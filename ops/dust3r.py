import sys, os
sys.path.append('./tools/dust3r')
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images, _resize_pil_image, ImgNorm, center_crop_pil_image
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
from dust3r.demo import get_3D_model_from_scene

import torch
from ops.cam_utils import Mcam
import copy
import numpy as np
from PIL import Image

from ops.utils.depth import refine_depth
import shutil
import tempfile

class Dust3rWrapper:
    def __init__(self, opts, load_mast3r=False, device='cuda'):
        self.opts = opts
        self.device = device
        self.backend = getattr(self.opts, 'backend', None) or ('mast3r' if load_mast3r else 'dust3r')

        # DUSt3R model (dense pointmap + global alignment)
        self.dust3r = load_model(self.opts.model_path, self.device)

        # MASt3R model (sparse global alignment) - only load when requested
        self.mast3r = None
        if self.backend == 'mast3r' or load_mast3r:
            from mast3r.model import AsymmetricMASt3R
            mast3r_path = getattr(
                self.opts,
                'mast3r_model_path',
                './tools/dust3r/checkpoint/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
            )
            self.mast3r = AsymmetricMASt3R.from_pretrained(mast3r_path).to(self.device)

    def run_dust3r_init(self, input_images = None, clean_pc = True, bg_mask=None): # setup self.scene
        if input_images is None:
            input_images = self.images

        # MASt3R backend: keep the same API but swap the reconstruction engine
        if getattr(self, 'backend', 'dust3r') == 'mast3r':
            scene = self.run_mast3r(input_images, clean_pc=clean_pc)
            self.scene = scene
            # provide a depth-like tensor for downstream heuristics (e.g., camera trajectory magnitude)
            if getattr(scene, 'depthmaps_dense', None) is not None and len(scene.depthmaps_dense) > 0:
                d = scene.depthmaps_dense[-1]
                if isinstance(d, torch.Tensor) and d.ndim == 1 and getattr(self, 'images', None):
                    H, W = self.images[-1]['img'].shape[2:]
                    if d.numel() == H * W:
                        d = d.view(H, W)
                self.depth = d.detach()
            elif getattr(scene, 'pts', None) is not None and len(scene.pts) > 0:
                # fallback: use point Z as a proxy
                self.depth = scene.pts[-1][..., 2].detach()
            return self.scene
            
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)

        mode = GlobalAlignerMode.PointCloudOptimizer 
        scene = global_aligner(output, device=self.device, mode=mode)
        
        if bg_mask is not None: # background added
            c2ws, c2w_bg =torch.eye(4).to(self.device),torch.eye(4).to(self.device)
            c2w_bg[2,3] = 1e-4
            scene.preset_pose([c2w_bg,c2ws])
            # scene.preset_focal([350.,350.])
            
        loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)
        
        if bg_mask is not None: 
            self.scale_depth(scene) # scale depth to [0.1, 0.4]
        self.depth=scene.get_depthmaps()[-1].detach()
        
        if bg_mask is not None: # background added
            dpt = refine_depth(self.depth.detach().cpu().numpy(),scene.get_depthmaps()[0].detach().cpu().numpy(),bg_mask)
            dpt = torch.tensor(dpt).to(self.device)
            scene._set_depthmap(0, dpt, force=True)
            
        if clean_pc:
            self.scene = scene.clean_pointcloud()
        else:
            self.scene = scene
        return self.scene

    def run_dust3r_preset(self, input_images, cams):
        from dust3r.cloud_opt import PointCloudOptimizer
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        dust3r_output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)

        view1, view2, pred1, pred2 = [dust3r_output[k] for k in 'view1 view2 pred1 pred2'.split()]
        scene = PointCloudOptimizer(view1, view2, pred1, pred2).to(self.device)

        f = cams[0].f
        f = f / cams[0].W * 512
        scene.preset_focal([f for _ in cams])

        # OpenGL cam to dust3r
        cams_rdf = [cam.copy() for cam in cams]
        for cam in cams_rdf:
            cam.R[[1,2],0] = -cam.R[[1,2],0]
            cam.R[0,1] = -cam.R[0,1]
            cam.R[0,2] = -cam.R[0,2]
            cam.T[[1,2]] = -cam.T[[1,2]]

        scene.preset_pose([cam.getC2W() for cam in cams_rdf])
        scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)
        return scene

    def get_scaled(self, scene):
        """
            匹配深度图的尺度
        """
        depth_base=scene.get_depthmaps()[0]
        print(depth_base.shape)
        m1 = torch.median(depth_base)
        m2 = torch.median(self.depth)
        return (m2/m1).cpu().item()

    def run_dust3r(self, input_images, clean_pc = True, cam_trajs=None): 
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)

        mode = GlobalAlignerMode.PointCloudOptimizer 
        scene = global_aligner(output, device=self.device, mode=mode)
        # if cam_trajs is not None:
        #     scene.preset_pose([cam.getC2W_RDF() for cam in cam_trajs])
        #     scene.preset_focal([cam.f for cam in cam_trajs])
        #     # scene.preset_principal_point([cam.c for cam in cam_trajs])
            
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)

        if clean_pc:
            scene = scene.clean_pointcloud()
            
        return scene

    def run_mast3r(self, input_images, clean_pc = True, cam_trajs=None): 
        pairs = make_pairs(input_images, scene_graph='logwin-3-noncyclic', prefilter=None, symmetrize=True)
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        filelist = [img['instance'] for img in input_images]
        with tempfile.TemporaryDirectory() as tmpdirname:
            cache_dir = tmpdirname
            if self.mast3r is None:
                raise RuntimeError("MASt3R backend requested but self.mast3r is not loaded. "
                                   "Set opt.dust3r.backend=mast3r (and provide mast3r_model_path if needed).")
            model = self.mast3r
            lr1 = float(getattr(self.opts, 'mast3r_lr1', 0.01))
            niter1 = int(getattr(self.opts, 'mast3r_niter1', 500))
            lr2 = float(getattr(self.opts, 'mast3r_lr2', 0.005))
            niter2 = int(getattr(self.opts, 'mast3r_niter2', 200))
            optim_level = str(getattr(self.opts, 'mast3r_optim_level', "refine+depth"))
            shared_intrinsics = bool(getattr(self.opts, 'mast3r_shared_intrinsics', True))
            matching_conf_thr = float(getattr(self.opts, 'mast3r_matching_conf_thr', 5.0))
            scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                        model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=self.device,
                                        opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                        matching_conf_thr=matching_conf_thr)
            # eliminate cache dependency: precompute dense points (+ depthmaps) inside tmpdir
            pts3d, depthmaps, _confs = scene.get_dense_pts3d(clean_depth=False)
            scene.pts = [p.detach() for p in pts3d]
            scene.depthmaps_dense = [d.detach() for d in depthmaps]
            # if cam_trajs is not None:
            #     scene.preset_pose([cam.getC2W_RDF() for cam in cam_trajs])
            #     scene.preset_focal([cam.f for cam in cam_trajs])
            #     # scene.preset_principal_point([cam.c for cam in cam_trajs])
        return scene


    def scale_depth(self, scene, target_min=0.1, target_max=0.4):
        # 应当废弃，但不能删，删会出问题
        depth_= scene.get_depthmaps()[-1]
        scale = (target_max-target_min) / (depth_.max()-depth_.min())
        for i,depth in enumerate(scene.get_depthmaps()):
            scaled_depth = (depth-depth.min()) * scale + target_min
            scene._set_depthmap(i, scaled_depth, force=True)
    
    def get_inital_pm(self):  # self.scene should have been setup
        if getattr(self, 'backend', 'dust3r') == 'mast3r':
            pts = self.get_pm_mast3r(self.scene, self.images)
        else:
            pts = self.get_pm(self.scene, self.images)
        self.pts = pts
        return pts

    def get_pm_mast3r(self, scene, images):
        # only mast3r use scene.pts to store the point cloud
        # Note that pts lies in world space already !!!!
        pts = scene.pts

        for i in range(len(pts)):
            pts_tmp = pts[i]  # [H,W,3]
            col = images[i]['img'].squeeze(0).permute(1,2,0).to(self.device)
            col = col * 0.5 + 0.5
            col = col.reshape(-1, 3)
            pts_tmp = torch.cat([pts_tmp, col], dim=-1) # [H, W, 6]
            # dust3r -> OpenGL
            pts_tmp[..., [1,2]] = -pts_tmp[..., [1,2]]
            pts[i] = pts_tmp
            
        return pts
    
    def get_pm(self, scene, images):
        '''
        return list of [H, W, 6] torch.tensor OpneGL
        '''
        pts = [i.detach() for i in scene.get_pts3d()] 

        for i in range(len(pts)):
            pts_tmp = pts[i]  # [H,W,3]
            col = images[i]['img'].squeeze(0).permute(1,2,0).to(self.device)
            col = col * 0.5 + 0.5
            pts_tmp = torch.cat([pts_tmp, col], dim=2) # [H, W, 6]
            # dust3r -> OpenGL
            pts_tmp[:,:, [1,2]] = -pts_tmp[:,:, [1,2]]
            pts[i] = pts_tmp
            
        return pts
    
    def get_cams_mast3r(self, scene=None):
        if scene is None:
            if getattr(self, 'scene', None) is None:
                raise ValueError("self.scene is not setup, pass a scene mannually")
            scene = self.scene
        cam_c2ws = scene.get_im_poses().detach()
        fs = scene.get_focals().detach()
        pps = scene.get_principal_points()
        # Prefer the actual processed image size if available
        if getattr(self, 'images', None) is not None and len(self.images) > 0:
            H, W = self.images[0]["img"].shape[2:]
            H, W = int(H), int(W)
        else:
            shape = scene.get_depthmaps()[0].shape
            H, W = int(shape[0]), int(shape[1])
        #cams = [Mcam().set_cam(W=W, H=H, c=tuple(pp), f=float(ff), R=cam[:3, :3], T=cam[:3, 3]) for cam,ff,pp in zip(cam_c2ws,fs,pps)]
        cams = [Mcam().set_cam(W=W, H=H, c=(pp[0].item(), pp[1].item()), f=ff.item(), R=cam[:3, :3], T=cam[:3, 3]) for cam,ff,pp in zip(cam_c2ws,fs,pps)]
        for cam in cams:
            cam.R[[1,2],0] = -cam.R[[1,2],0]
            cam.R[0,1] = -cam.R[0,1]
            cam.R[0,2] = -cam.R[0,2]
            cam.T[[1,2]] = -cam.T[[1,2]]
        return cams
    
    
    def get_cams(self, scene=None):
        if getattr(self, 'backend', 'dust3r') == 'mast3r':
            return self.get_cams_mast3r(scene=scene)

        if scene is None:
            if getattr(self, 'scene', None) is None:
                raise ValueError("self.scene is not setup, pass a scene mannually")
            scene = self.scene
        cam_c2ws = scene.get_im_poses().detach()
        fs = scene.get_focals().detach()
        # erase dependency on self.images
        shape = scene.get_depthmaps()[0].shape
        #shape = self.images[0]['true_shape']
        H, W = int(shape[0]), int(shape[1])
        cams = [Mcam().set_cam(W=W, H=H, c=(W//2, H//2), f=ff.item(), R=cam[:3, :3], T=cam[:3, 3]) for cam,ff in zip(cam_c2ws,fs)]
        for cam in cams:
            cam.R[[1,2],0] = -cam.R[[1,2],0]
            cam.R[0,1] = -cam.R[0,1]
            cam.R[0,2] = -cam.R[0,2]
            cam.T[[1,2]] = -cam.T[[1,2]]
        return cams

    def get_scene_info(self, scene ,ref_images):
        c2ws = scene.get_im_poses().detach()
        principal_points = scene.get_principal_points().detach()
        focals = scene.get_focals().detach()
        pcds = [i.detach() for i in scene.get_pts3d()]
        
        cams = []
        for f, c2w, c in zip(focals, c2ws, principal_points):
            f = f.item()
            c = (c[0].item(), c[1].item())
            R, T = c2w[:3, :3], c2w[:3, 3:].squeeze()
            R = torch.stack([R[:, 0], -R[:, 1], -R[:, 2]], 1) # from RDF to RUB for Rotation
            def npy(x):
                return x.cpu().numpy()
            cam = Mcam().set_cam(c=c, f=f, R=npy(R), T=npy(T))
            cams.append(cam)
        
        ## masks for cleaner point cloud
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
        masks = scene.get_masks()
        depth = scene.get_depthmaps()
        bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
        masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
        masks = to_numpy(masks_new)
        
        for i in range(len(pcds)):
            pts = pcds[i]  # [H,W,3]
            col = ref_images[i].to(pts.device)
            print(col.shape)
            pts = torch.cat([pts, col], dim=2) # [H, W, 6]
            pts[:,:, [1,2]] = -pts[:,:, [1,2]]
            pcds[i] = pts
        
        return pcds, cams, masks
        

    def _load_images(self, image_dir,opt=None):
        # [!] img range here is [-1,1], save_image need [0,1]
        ## load images
        ## dict_keys(['img', 'true_shape', 'idx', 'instance', 'img_ori']),张量形式
        if isinstance(image_dir, str):
            image_dir=[image_dir]
        if opt is None:
            images = load_images(image_dir, size=512,force_1024 = True)
        else:
            images = load_images(image_dir, size=opt.dust3r.size,square_ok=opt.dust3r.square_ok,force_1024 = opt.dust3r.force_1024)
            
        img_ori = (images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [576,1024,3] [0,1]

        if len(images) == 1:
            images = [images[0], copy.deepcopy(images[0])]
            images[1]['idx'] = 1
            # IMPORTANT: 'instance' is used as a stable identifier (e.g., MASt3R caching / pair naming).
            # Make sure the duplicated image has a different instance id.
            images[1]['instance'] = str(images[1]['idx'])
            
        return images, img_ori

    def load_initial_images(self, image_dir,opt=None):
        # [!] img range here is [-1,1], save_image need [0,1]
        ## load images
        ## dict_keys(['img', 'true_shape', 'idx', 'instance', 'img_ori']),张量形式
        images, img_ori=self._load_images(image_dir,opt)
        self.images = images
        return images, img_ori
    
    def _load_our_images(self, imgs, size=512):
        ''' imgs: [N, H, W, 3] np.ndarray in range [0, 1]
        size: 512 or 224
        ret: imgs[i]["img"] = [3, H, W] torch.tensor ranged in [-1, 1]
        '''
        if isinstance(imgs, np.ndarray):
            pass
        elif isinstance(imgs, torch.Tensor):
            imgs = imgs.cpu().numpy()
        elif isinstance(imgs, str):
            imgs = np.array(Image.open(imgs).convert('RGB')) / 255.0
        elif isinstance(imgs, list) and isinstance(imgs[0], np.ndarray):
            imgs = np.stack(imgs, axis=0)

        if imgs.max() > 10.0:
            raise ValueError("Image range should be [0, 1]")

        if imgs.ndim == 3:
            print("dust3r: single image auto repeat")
            imgs = imgs[None,:].repeat(2, axis=0)
            print(imgs.shape)
        
        res = []
        for img in imgs:
            img = Image.fromarray((img*255.0).astype(np.uint8))
            img_ori = img
            img = center_crop_pil_image(img)
            W1, H1 = img.size
            if size == 224:
                # resize short side to 224 (then crop)
                img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
            else:
                # resize long side to 512
                img = _resize_pil_image(img, size)
            W, H = img.size
            cx, cy = W//2, H//2
            if size == 224:
                half = min(cx, cy)
                img = img.crop((cx-half, cy-half, cx+half, cy+half))
            else:
                halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
                img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

            W2, H2 = img.size
            print(f' - adding img with resolution {W1}x{H1} --> {W2}x{H2}')
            res.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
                [img.size[::-1]]), idx=len(res), instance=str(len(res)), img_ori=ImgNorm(img_ori)[None], ))

        print(f' (Found {len(res)} images)')
        return res
        