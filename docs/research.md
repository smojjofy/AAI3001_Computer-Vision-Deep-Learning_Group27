# Constructing NeRFs and 3D Gaussian Splats from Visual Data

## Executive summary

Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS) solve the same broad problem: infer a view-dependent, renderable representation of a static scene from a set of photographs or video frames. Neither method normally needs a ground-truth mesh, dense 3D labels, or per-pixel depth. Both instead minimize image-reconstruction error: they render a training camera view from a learned scene representation and compare the result with the captured image.

The most important practical fact is that the visual data is not just a folder of images. A successful reconstruction also needs a consistent camera model: image dimensions, focal lengths, principal point, distortion treatment, and one pose per image. The original NeRF assumes these camera parameters are known; the canonical 3DGS pipeline uses the same kind of camera calibration and additionally consumes the sparse 3D points produced by Structure-from-Motion (SfM). COLMAP is the common bridge from unposed images to these inputs.[^1][^2]

The methods differ after camera preparation:

| Dimension | NeRF | 3D Gaussian Splatting |
|---|---|---|
| Native scene representation | An implicit continuous function stored in an MLP | An explicit set of 3D Gaussian primitives |
| Typical initialization | Random network weights; no point cloud required | Sparse SfM points become Gaussian centers |
| Appearance | MLP predicts color as a function of position and viewing direction | Per-Gaussian spherical-harmonic coefficients model view dependence |
| Rendering | Sample many points along each ray and numerically integrate density and color | Project Gaussians to screen-space ellipses, sort, and alpha-composite |
| Capacity control | Hierarchical ray sampling; modern variants add proposal networks or grids | Periodic cloning, splitting, and pruning of Gaussians |
| Geometry output | Density field; mesh extraction is a separate post-process | Point/splat field; a clean watertight mesh is not the native output |
| Main strength | Flexible continuous volumetric model and mature view-synthesis quality | Fast training and real-time rendering with explicit editable primitives |
| Main weakness | Expensive ray marching and network evaluation | Memory growth, heuristic densification, and sensitivity to initialization/capture quality |

The shared bottleneck is observability. A point or surface must be seen in multiple sufficiently different views with enough texture and stable appearance to constrain its location. More images do not compensate for severe blur, motion, pure camera rotation, changing geometry, or incompatible exposure. Good capture design usually improves both methods more than changing a late-stage hyperparameter.[^2][^3]

## 1. What is being reconstructed?

### 1.1 Novel-view synthesis, not necessarily a conventional 3D model

The target is a function that can predict the color of a pixel for a new camera pose. This is stricter than stitching photographs but less restrictive than recovering a physically exact surface. A model can reproduce the observed images while containing ambiguous or view-dependent geometry, especially in regions that are poorly observed.

NeRF represents a static scene as a continuous 5D radiance field. Given a 3D location \(\mathbf{x}=(x,y,z)\) and viewing direction \(\mathbf{d}\), its network predicts a volume density \(\sigma\) and emitted RGB radiance \(\mathbf{c}\):

\[
F_\theta(\mathbf{x},\mathbf{d})\rightarrow (\mathbf{c},\sigma).
\]

Density is a differential opacity: it controls how likely a ray is to terminate around a location. Color is allowed to depend on direction so that the model can represent effects such as specular highlights that change with viewpoint. The original NeRF architecture restricts density to depend only on position and predicts color from both a learned position feature and the viewing direction, which encourages multi-view-consistent geometry while retaining view-dependent appearance.[^4]

3DGS replaces the single implicit function with a collection of explicit volumetric primitives. Each primitive has a center \(\boldsymbol{\mu}_i\), positive-definite covariance \(\boldsymbol{\Sigma}_i\), opacity \(\alpha_i\), and appearance coefficients. A Gaussian is not just a colored point: its covariance lets it become elongated and oriented, so a surface patch can be approximated by a thin anisotropic ellipsoid rather than by many isotropic points.[^5]

### 1.2 The inverse-rendering loop

Both approaches can be summarized as:

```text
RGB images / video
        |
        v
frame selection, resizing, undistortion, masking
        |
        v
camera intrinsics + camera-to-world poses (+ sparse SfM points)
        |
        +------------------------+
        |                        |
        v                        v
NeRF field initialization   Gaussian initialization
        |                        |
        v                        v
ray samples -> volume render  project -> sort -> alpha composite
        |                        |
        +----------+-------------+
                   v
        compare rendered pixels with captured pixels
                   |
                   v
              backpropagation
                   |
                   v
       update representation and repeat
```

The rendered training image is the supervision signal. This is why the methods are often called self-supervised or photometrically supervised: the image itself supplies the target, while differentiable rendering supplies the connection back to the scene parameters. The supervision is not truly unconstrained; camera calibration, scene assumptions, and the coverage of the images determine what can be learned.

## 2. The shared visual-data front end

### 2.1 Capture design

For a static object or room, capture a path that provides both overlap and parallax:

- Keep the scene and camera exposure stable. Do not move objects between frames.
- Use neighboring views with substantial visual overlap, but also move the camera in 3D rather than only rotating it.
- Ensure important surfaces appear in at least several images. COLMAP recommends that each object be visible in at least three images as a practical minimum; additional views are useful when they add new geometry or reduce occlusion.[^2]
- Avoid motion blur, severe compression, extreme exposure changes, direct sun-to-shadow transitions, and strong reflections where possible.
- Keep focus and focal length fixed if possible. A zoom or focus change can violate the shared-intrinsics assumption unless the pipeline models it.
- For video, extract frames at a rate that preserves useful viewpoint change without creating hundreds of nearly duplicate images. Near-duplicates increase computation without necessarily adding geometric constraints.

The goal is not uniform angular sampling alone. A camera that rotates in place sees a different direction but supplies almost no triangulation baseline, so depth is fundamentally weak. NeRF may still synthesize plausible views in this case, but its depth can be poor; SfM can fail to initialize because the epipolar geometry is degenerate.[^6]

### 2.2 Image preprocessing

Typical preprocessing consists of:

1. Extracting frames from video and removing blurry, duplicate, or badly exposed frames.
2. Keeping image-to-file naming stable and unique.
3. Resizing images consistently; when an image is resized by a factor \(s\), focal lengths and principal point coordinates must be scaled by the same factor.
4. Estimating or applying lens distortion correction. Distorted pixels should not be passed to a pinhole renderer with uncorrected intrinsics.
5. Optionally creating masks for moving people, cars, turntables, tripods, or background regions that should not be fitted.
6. Converting color and alpha consistently. A mismatch between linear and display-referred RGB, or between premultiplied and straight alpha, can appear as a persistent reconstruction error.

If images were captured with changing exposure, the model may use geometry, opacity, or view-dependent color to explain brightness differences. This can create incorrect geometry or shimmering. Exposure compensation and robust losses are therefore useful extensions, but consistent capture remains preferable. The original 3DGS implementation has since added exposure-compensation support, illustrating that photometric inconsistency is a practical issue rather than a minor detail.[^7]

### 2.3 Camera intrinsics and extrinsics

For a perspective camera, the intrinsic matrix is commonly written as

\[
K=\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix},
\]

where \(f_x,f_y\) are focal lengths in pixels and \((c_x,c_y)\) is the principal point. Distortion parameters may add radial and tangential terms. The extrinsics specify the camera's rotation and translation relative to a shared world frame. A renderer needs these parameters to turn pixel coordinates into rays or to project 3D primitives into the image.

The common unposed-image route is SfM:

1. Detect local image features and descriptors.
2. Match features between overlapping image pairs.
3. Geometrically verify matches with epipolar constraints and robust estimation.
4. Seed a two-view reconstruction.
5. Register more images, triangulate 3D points, reject outliers, and refine camera and point parameters with bundle adjustment.

This is the incremental pipeline described by COLMAP's authors. COLMAP's output includes camera intrinsics, image poses, 2D observations, and sparse 3D points. Its database distinguishes camera-level intrinsics from image-level extrinsics: multiple images can share one camera model while each image has its own pose.[^3][^8]

Alternative sources of poses include factory calibration, a calibrated multi-camera rig, visual-inertial SLAM, LiDAR/AR tracking, or a photogrammetry application such as Metashape or RealityCapture. A LiDAR or SLAM pose can be faster or more reliable for low-texture sequences, while SfM can be more accurate when the images have strong static features and sufficient overlap.[^9]

### 2.4 Coordinate conventions and scale

Coordinate mistakes are among the most common causes of an apparently broken reconstruction. A dataset may store world-to-camera transforms while a renderer expects camera-to-world transforms. It may also use OpenCV axes while the renderer uses OpenGL/Blender axes. The original NeRF code uses camera-to-world matrices and an OpenGL-style camera convention; COLMAP stores image poses as world-to-camera quaternion-plus-translation records. These must be converted deliberately, not merely transposed.[^4][^8]

Monocular SfM normally recovers structure up to a global similarity transform: rotation, translation, and scale are not all metrically observable from images alone. This does not prevent novel-view synthesis because the cameras and scene share the same coordinate system, but it matters if the output must be measured in meters or combined with a physical object. A known distance, depth sensor, calibrated rig, or external tracking system is needed to establish metric scale.

### 2.5 What COLMAP produces for downstream methods

A useful reconstruction-ready dataset contains:

```text
images/
  image_0001.jpg
  image_0002.jpg
  ...
sparse/0/
  cameras.bin or cameras.txt
  images.bin or images.txt
  points3D.bin or points3D.txt
```

For NeRF, the essential outputs are image pixels, intrinsics, poses, and near/far bounds or an equivalent scene box. For the canonical 3DGS pipeline, the sparse point cloud is also important because it seeds the initial Gaussian centers and colors. COLMAP's dense MVS output is not required by the original 3DGS method; a sparse SfM point cloud is enough to initialize the explicit primitives, which are then densified during training.[^5][^7]

## 3. Constructing a NeRF

### 3.1 Ray construction

For a pixel \(\mathbf{p}=(u,v)\), unproject the pixel through the camera intrinsics to obtain a camera-space direction, rotate it into world space, and set the ray origin to the camera center. In simplified notation:

\[
\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}, \qquad t\in[t_n,t_f],
\]

where \(\mathbf{o}\) is the camera center, \(\mathbf{d}\) is the normalized world-space direction, and \([t_n,t_f]\) brackets the relevant scene content. The same operation is repeated for the pixels selected in a training batch.

The near and far bounds are not cosmetic. If the interval excludes geometry, the field cannot explain the pixel correctly. If it is excessively large, most samples fall in empty space and optimization becomes less efficient. Forward-facing NeRF pipelines often use normalized device coordinates; 360-degree and unbounded variants use different scene parameterizations.

### 3.2 Positional encoding

A plain MLP tends to learn low-frequency functions first and can oversmooth thin geometry and high-frequency texture. NeRF maps each scalar coordinate through Fourier features:

\[
\gamma(p)=\left(\sin(2^0\pi p),\cos(2^0\pi p),\ldots,
\sin(2^{L-1}\pi p),\cos(2^{L-1}\pi p)\right).
\]

The original implementation applies this encoding independently to position and viewing direction, using \(L=10\) for position and \(L=4\) for direction. The encoded position is passed through the main MLP. The network produces density and a learned feature; the feature is combined with encoded direction to produce view-dependent RGB.[^4]

Positional encoding is a representation choice, not a preprocessing step performed on the photographs. It changes the coordinate basis in which the MLP is optimized. Modern systems may replace the Fourier basis with a multiresolution hash grid, voxel features, or other encodings. Instant-NGP, for example, pairs a small MLP with trainable multiresolution hash tables and fused GPU kernels, reducing the cost of evaluating a field while preserving high-frequency detail.[^10]

### 3.3 Stratified volume sampling

For each ray, the original NeRF divides \([t_n,t_f]\) into \(N_c\) bins and samples one random point in each bin. At a sampled distance \(t_i\), the field returns \((\mathbf{c}_i,\sigma_i)\). With \(\delta_i=t_{i+1}-t_i\), define:

\[
\alpha_i=1-\exp(-\sigma_i\delta_i),
\]

\[
T_i=\exp\left(-\sum_{j<i}\sigma_j\delta_j\right),
\qquad
w_i=T_i\alpha_i.
\]

The rendered ray color is:

\[
\hat{\mathbf{C}}(\mathbf{r})=\sum_i w_i\mathbf{c}_i.
\]

This is discretized emission-absorption volume rendering. It is differentiable with respect to the colors, densities, sample locations, camera parameters, and network weights. The transmittance term makes samples behind an opaque region contribute little, while the density-to-alpha conversion makes the result equivalent to front-to-back alpha compositing.[^4]

### 3.4 Hierarchical sampling

Uniform sampling wastes queries in free space and behind already opaque surfaces. NeRF therefore uses two networks or two passes:

1. A coarse network evaluates \(N_c\) stratified samples.
2. The coarse weights \(w_i\) form a piecewise-constant probability distribution along the ray.
3. Additional samples are drawn from this distribution using inverse-transform sampling.
4. A fine network evaluates the union of coarse and newly sampled locations.
5. Both coarse and fine renderings receive an image loss, while the fine result is used as the final prediction.

The original paper uses batches of 4096 rays, 64 coarse samples, and 128 additional fine samples as a reference configuration. These numbers are not universal; the correct choice depends on scene scale, resolution, and the chosen NeRF variant. The key idea is adaptive allocation of field queries toward regions likely to affect the pixel.[^4]

### 3.5 Scene-specific optimization

The original NeRF trains a separate network for each scene. A typical iteration is:

```text
select random training images and pixels
construct rays using their current/fixed camera parameters
sample coarse points along rays
query coarse field and volume-render
sample additional points from coarse weights
query fine field and volume-render
compute image loss against captured pixels
backpropagate through rendering and the MLP
update network weights
```

The baseline objective is the sum of squared RGB error for coarse and fine renderings:

\[
\mathcal{L}_{\text{rgb}}=
\sum_{\mathbf{r}}
\left\|\hat{\mathbf{C}}_c(\mathbf{r})-\mathbf{C}^{*}(\mathbf{r})\right\|_2^2+
\left\|\hat{\mathbf{C}}_f(\mathbf{r})-\mathbf{C}^{*}(\mathbf{r})\right\|_2^2.
\]

The original NeRF implementation reports roughly 100k–300k iterations and one to two days on a V100 for a single scene under its 2020 configuration. This is historically informative rather than a current hardware promise: multiresolution grids, fused kernels, occupancy structures, proposal networks, mixed precision, and optimized frameworks can make later NeRF-family implementations much faster.[^4][^10]

### 3.6 NeRF variants that change construction

The NeRF name now covers a family of field representations. The most relevant changes to construction from visual data are:

**Mip-NeRF.** Instead of treating each pixel as an infinitesimal ray, cone tracing represents the pixel footprint as a conical frustum. Integrated positional encoding averages Fourier features over that region, suppressing frequencies that would alias at the target scale. A single MLP can then represent multiple scales more naturally.[^11]

**Mip-NeRF 360.** Unbounded outdoor or room-scale scenes contain both nearby detail and content at very large distances. Mip-NeRF 360 uses a nonlinear contraction of unbounded space, online distillation from a larger radiance-field MLP to a proposal MLP, and a distortion regularizer that discourages dispersed or ambiguous ray weights. These changes address scale imbalance, expensive sampling, and floaters in 360-degree captures.[^12]

**Hash-grid and grid-backed fields.** Instant-NGP and related systems move more capacity from the MLP into a learned spatial encoding. The field is still optimized through differentiable rendering, but the representation has a faster lookup structure and a smaller network.[^10]

**Pose-optimized NeRFs.** The original method assumes known poses. NeRF-- demonstrates that camera intrinsics and extrinsics can be parameterized as trainable variables and jointly optimized with the field through the same photometric loss. This can recover useful poses without an SfM preprocessing stage, but the coupled problem has local minima and ambiguities, especially between focal length and scene/camera scale.[^13]

### 3.7 Extracting geometry from a NeRF

A NeRF's native output is not a mesh. To obtain a surface, evaluate density on a 3D grid, select an isovalue, and run an algorithm such as Marching Cubes. The resulting surface may require smoothing, hole filling, pruning, and texture handling. Density is optimized for rendering, not necessarily as a calibrated signed distance function, so the extracted mesh can contain floaters, thick surfaces, or ambiguous backgrounds. The original reference repository includes mesh extraction as a separate example rather than as part of the core training loop.[^4]

## 4. Constructing 3D Gaussian Splats

### 4.1 SfM initialization is part of the model construction

The canonical 3DGS system begins with a sparse point cloud produced during camera calibration. This is the major difference from a randomly initialized NeRF MLP. The point cloud provides approximate scene support and a coordinate frame; 3DGS then learns how many primitives are needed and how they should expand or contract.[^5]

If a COLMAP point cloud is unavailable, the reference implementation can create random points for synthetic NeRF data inside known scene bounds. That fallback is useful for controlled synthetic scenes but is not generally equivalent to having accurate multi-view geometry for a real capture.[^14]

### 4.2 Gaussian parameterization

The scene is represented by:

\[
\mathcal{G}=\{(\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i,\alpha_i,\mathbf{h}_i)\}_{i=1}^{N},
\]

where \(\mathbf{h}_i\) denotes appearance coefficients. The covariance is parameterized by a scale vector \(\mathbf{s}_i\) and a rotation quaternion \(\mathbf{q}_i\):

\[
\boldsymbol{\Sigma}_i=R(\mathbf{q}_i)
\operatorname{diag}(\mathbf{s}_i^2)
R(\mathbf{q}_i)^T.
\]

This factorization guarantees a valid positive-semidefinite covariance when the scales are positive and the rotation is normalized. In the official implementation, scales are stored in log form and activated with an exponential; opacity is stored in logit form and activated with a sigmoid; the quaternion is normalized before use.[^15]

The reference initialization from a point cloud is concrete:

- Set each Gaussian center to an SfM point.
- Convert its RGB color to the zero-order spherical-harmonic coefficient; initialize higher-order coefficients to zero.
- Estimate an initial isotropic scale from local point spacing using a CUDA nearest-neighbor distance routine, then repeat that scale across the three axes.
- Initialize rotations to identity.
- Initialize opacity to a low value, 0.1 in the current reference code, represented internally by its inverse sigmoid.

This initialization gives the optimizer small, translucent primitives distributed near observed geometry. The scales and rotations then become learnable, allowing a Gaussian to align with a surface or elongate across a region where projected image evidence demands it.[^15]

### 4.3 Projection to screen space

For a selected camera, transform a Gaussian center into camera coordinates and project it to a 2D mean. The 3D covariance is locally transformed by the camera projection Jacobian, producing a 2D covariance that describes an ellipse on the image plane. In simplified form:

\[
\boldsymbol{\Sigma}'_i \approx J_i W\boldsymbol{\Sigma}_i W^T J_i^T,
\]

where \(W\) is the world-to-camera linear transform and \(J_i\) is the Jacobian of perspective projection at the Gaussian center. The projected Gaussian contributes to a pixel \(\mathbf{u}\) according to a 2D Gaussian kernel:

\[
g_i(\mathbf{u})=
\exp\left(-\frac{1}{2}
(\mathbf{u}-\boldsymbol{\mu}'_i)^T
\boldsymbol{\Sigma}_i'^{-1}
(\mathbf{u}-\boldsymbol{\mu}'_i)\right).
\]

Its effective pixel alpha is the learned opacity multiplied by this kernel. In practice the renderer also culls primitives outside the view frustum or with zero projected radius.

### 4.4 View-dependent appearance with spherical harmonics

Each Gaussian stores a compact angular function rather than one fixed RGB color. Spherical-harmonic (SH) coefficients are evaluated at the direction from the Gaussian toward the camera. The zero-order coefficient acts like a diffuse/base color, while higher orders model smooth view-dependent changes.

The official implementation progressively increases the active SH degree every 1000 iterations until the configured maximum. This delays high-frequency angular variation while centers, opacity, and low-order color stabilize. The default maximum degree is 3 in the reference training interface.[^7][^16]

SH is a useful approximation to smooth view dependence, not a complete physical reflectance model. Highly specular, transparent, refractive, or rapidly changing effects can still be difficult. A splat may reproduce those effects from the observed camera range without correctly representing the underlying material outside that range.

### 4.5 Visibility-aware tile rasterization

The central rendering optimization is to use GPU-friendly rasterization rather than repeatedly evaluating a neural network at many samples along every ray. The screen is divided into tiles. Each Gaussian is duplicated into the list of tiles overlapped by its projected bounding rectangle, then assigned a key combining tile ID and depth. A GPU radix sort creates a list ordered first by tile and then by depth. Each tile can then blend its relevant Gaussians in parallel.[^17]

For a pixel, the renderer composites front-to-back:

\[
\hat{\mathbf{C}}(\mathbf{u})=
\sum_i T_i\,a_i(\mathbf{u})\,\mathbf{c}_i(\mathbf{d}),
\qquad
T_i=\prod_{j<i}(1-a_j(\mathbf{u})),
\]

where \(a_i(\mathbf{u})\) is the Gaussian's effective screen-space alpha and \(\mathbf{c}_i(\mathbf{d})\) is its SH-evaluated color. This resembles the discretized NeRF equation, but the samples are explicit projected primitives rather than points sampled along every ray. The approximation to exact volumetric ordering is a deliberate tradeoff: tile-level depth sorting makes the operation fast and differentiable enough for training while avoiding a separate sort for every pixel.[^17]

### 4.6 Photometric optimization

The training loop selects a camera, renders its image, compares the render with the captured image, and backpropagates into Gaussian parameters. The reference code uses a mixture of pixelwise L1 error and a differentiable SSIM term:

\[
\mathcal{L}_{\text{rgb}}=
(1-\lambda)\mathcal{L}_1+
\lambda(1-\operatorname{SSIM}).
\]

The gradients update centers, scales, rotations, opacity, SH coefficients, and, where enabled, exposure parameters. The official repository exposes the same training model through `python train.py -s <dataset>` and supports COLMAP or NeRF-synthetic dataset layouts.[^7]

Unlike NeRF, the number of Gaussians is not fixed for the whole run. The representation is optimized and structurally edited in an interleaved loop.

### 4.7 Adaptive density control: clone, split, prune

The reference 3DGS density-control mechanism accumulates the magnitude of the image-space gradient of projected Gaussian positions for visible primitives. A high average gradient indicates that the current primitive placement cannot explain image evidence well; more capacity may be needed there.[^18]

At periodic intervals:

1. **Clone small, high-gradient Gaussians.** A small primitive with a large positional gradient is copied so that nearby capacity can move toward missing detail.
2. **Split large, high-gradient Gaussians.** A large primitive that spans incompatible detail is replaced by smaller children sampled around the parent, allowing finer spatial structure.
3. **Prune weak or oversized Gaussians.** Low-opacity primitives are removed. After the early stage, primitives with excessive screen-space radius or world-space scale can also be removed.
4. **Reset opacity periodically.** The reference code resets opacities to a low value every 3000 iterations, allowing ineffective primitives to become pruneable and preventing early opacity decisions from becoming permanent.

The reference defaults are informative: densification begins at iteration 500, stops at 15,000, runs every 100 iterations, uses a positional-gradient threshold of 0.0002, prunes below opacity 0.005, and resets opacity every 3000 iterations. These are implementation defaults, not scene-independent laws. Resolution, image normalization, scene extent, point-cloud density, and camera quality all affect the gradient scale and the correct thresholds.[^7][^18]

The same mechanism is also a major source of 3DGS behavior. Under-densification produces holes, blur, and missing thin structures. Over-densification produces large files, redundant overlapping splats, cloudy backgrounds, or unstable memory use. Subsequent research has proposed error-driven density criteria, explicit point-count budgets, improved opacity handling, and more structure-aware pruning; these improve the original heuristic but do not remove the need for good visual input.[^19]

### 4.8 What a trained splat is—and is not

A trained splat is an explicit, renderable collection of Gaussian primitives. It is easy to inspect, edit, cull, transform, and render with a specialized viewer. However, it is not automatically a polygon mesh, a watertight surface, or a physically correct decomposition into objects and materials. Splats can overlap, float, extend through surfaces, and encode appearance in view-dependent coefficients. Mesh extraction or surface regularization is a separate task.

## 5. Why the same images affect the two methods differently

### 5.1 Pose quality

NeRF can absorb some calibration error by producing a blurred or view-dependent field, but pose errors typically create floating geometry, ghosting, and loss of sharpness. 3DGS is even more visibly tied to initialization because its centers begin at SfM points and its projected gradients drive densification. A bad sparse cloud can seed primitives in incorrect locations; later optimization may partially repair this but can also densify the error.

The practical response is to inspect the sparse reconstruction before training: camera trajectory, registered-image count, reprojection residuals, point-cloud extent, and point distribution. If cameras cluster on one side or the cloud contains disconnected components, training is unlikely to fix the underlying geometry.

### 5.2 Texture and parallax

SfM depends on recognizable correspondences. A textureless wall, repeated pattern, transparent object, or glossy surface may not produce stable features. NeRF does not require a point cloud, but it still needs multi-view image evidence to decide where density belongs. When all views agree only on color but not depth, the loss admits many explanations.

Parallax is the primary geometric signal. Pure rotation can support image-direction interpolation but supplies little depth. Very small baselines make depth numerically sensitive; very large viewpoint jumps reduce feature overlap and increase occlusion. A good capture path balances both.

### 5.3 Dynamic content

Both canonical methods assume a static scene. If a person, fan, plant, vehicle, or screen changes between views, one static field is asked to explain multiple incompatible states. The result can be duplicated limbs, blurred moving objects, transparent-looking geometry, or time-specific artifacts. Masking, selecting a static interval, using a dynamic-scene method, or jointly modeling time is required when motion is material.

### 5.4 View-dependent and non-Lambertian appearance

Direction-dependent color is an explicit design feature of NeRF and 3DGS, so neither requires all surfaces to be Lambertian. Nevertheless, view dependence is underconstrained outside observed directions. A highlight that moves across a surface can be encoded as angular color, incorrect geometry, or a mixture of both. Strong reflection and refraction remain difficult because the scene appearance may depend on unobserved environment lighting and complex light transport.

### 5.5 Occlusion and coverage

No method can reconstruct an unseen surface from ordinary multi-view images without a learned prior or additional sensor. Back sides, undersides, interiors, and regions behind permanent occluders remain ambiguous. A 360-degree capture reduces this problem for closed objects and rooms, but it does not guarantee complete coverage: the camera must still see the relevant surfaces at useful distances and angles.

## 6. A reproducible construction workflow

### 6.1 Recommended baseline for a real object or scene

1. **Plan the capture.** Decide the intended novel-view range, then capture overlapping views around the target with real camera translation. Keep lighting, focus, zoom, and scene state stable.
2. **Select frames.** Remove blur, duplicate frames, severe exposure changes, and moving-object contamination. Preserve enough overlap for matching.
3. **Calibrate or run SfM.** Use known intrinsics when available; otherwise run COLMAP feature extraction, matching, geometric verification, reconstruction, and bundle adjustment.
4. **Inspect calibration.** Verify that most images register, camera centers form the intended path, the sparse points occupy the target rather than the background, and no large disconnected component dominates.
5. **Undistort and resize.** Use one consistent image set and update intrinsics whenever dimensions change.
6. **Normalize the scene.** Convert all poses to the renderer's convention, center and scale the coordinate system, and define a scene box or extent. Preserve the transform so output can later be mapped back to the capture frame.
7. **Train a baseline.** Use the same held-out images for evaluation. Start at a moderate resolution, then increase resolution only after pose and coverage are validated.
8. **Inspect intermediate views.** Render training cameras, held-out cameras, camera trajectories, depth, opacity/accumulation, and—during 3DGS training—splat count and scale distributions.
9. **Tune the representation.** For NeRF, adjust sampling, occupancy/proposal settings, encoding, and scene bounds. For 3DGS, adjust image resolution, densification threshold, densification interval, maximum point count, opacity reset behavior, and pruning.
10. **Export with expectations set correctly.** A NeRF checkpoint is a field plus camera/dataset metadata; a splat is a Gaussian asset plus appearance coefficients. Mesh export requires an additional geometry-processing stage.

### 6.2 Practical tool paths

The original NeRF repository uses LLFF/COLMAP preprocessing for real images, then trains and renders a scene-specific network. It also includes an example of Marching Cubes mesh extraction.[^4]

The original 3DGS repository provides a script for making optimization-ready SfM datasets, a PyTorch/CUDA optimizer, a network viewer, and a real-time OpenGL viewer. Its documented command is:

```bash
python train.py -s <path-to-colmap-or-nerf-synthetic-dataset>
```

The reference implementation defaults to 30,000 iterations and provides options for evaluation splits, image resolution, SH degree, densification, pruning, opacity reset, and checkpointing.[^7]

Nerfstudio packages much of the data preparation into `ns-process-data`. For ordinary image folders or video, it invokes FFmpeg and COLMAP to create a dataset with camera poses; for supported LiDAR, SLAM, or photogrammetry exports, it can consume externally computed poses. Its documented assumptions explicitly include known poses, static objects, constant appearance, and dense enough coverage.[^9][^20]

### 6.3 Data and experiment bookkeeping

For each reconstruction, record:

- original image files and frame-selection rules;
- camera model, intrinsics, distortion parameters, and calibration source;
- image resize/undistortion transform;
- pose convention and world-to-camera/camera-to-world conversion;
- similarity normalization and metric-scale reference, if one exists;
- train/validation/test split;
- model, renderer, resolution, iteration, and density-control settings;
- software versions and GPU/precision settings;
- final primitive count, checkpoint size, and evaluation metrics.

This is essential because a visually pleasing render can be produced by a representation that is badly scaled, overfit to training views, or incompatible with another coordinate system.

## 7. Evaluation and diagnosis

### 7.1 Novel-view metrics

Render held-out images from their original camera poses and compare them with the captured targets. Common metrics are:

- **PSNR:** pixelwise logarithmic fidelity; useful but sensitive to small alignment and exposure differences.
- **SSIM:** structural similarity; often better aligned with local contrast and structure.
- **LPIPS:** learned perceptual distance; useful for perceptual differences but not a geometry guarantee.

The 3DGS reference repository computes L1 and PSNR at configured test iterations and exposes separate rendering and metrics scripts.[^7] Metrics should be accompanied by visual inspection because a model can achieve good average RGB error while producing bad depth, floaters, or poor camera-path behavior.

### 7.2 Geometry evaluation

If depth or geometry matters, evaluate more than RGB:

- render depth maps and compare against LiDAR, MVS, or synthetic ground truth after scale alignment;
- measure camera trajectory error when ground truth is available;
- inspect cross-sections for floaters and overly thick surfaces;
- evaluate completeness and accuracy separately, since unseen regions may be absent rather than wrong;
- test a novel camera path rather than only interpolated training viewpoints.

Do not interpret a NeRF density field or Gaussian center cloud as a metric surface without checking how it was produced. Both are optimized primarily for image formation.

### 7.3 Failure-signature table

| Symptom | Likely cause | First checks | Typical remedy |
|---|---|---|---|
| COLMAP registers few images | Blur, low texture, insufficient overlap, wrong matching strategy | Match graph and registered-camera count | Capture sharper, more overlapping views; add texture; use stronger matching or external poses |
| Whole render is warped or duplicated | Wrong pose convention, wrong intrinsics, mismatched image names | Project sparse points back into input images | Fix transform direction/axes and intrinsics; keep image ordering exact |
| Good training views but bad held-out views | Sparse coverage or overfitting | Compare train versus held-out renderings | Add views, improve camera path, use regularization or a better bounded/unbounded variant |
| Blurry details | Under-sampling, low resolution, pose noise, insufficient high-frequency capacity | Render depth/weights and inspect sharpness across views | Improve poses, use positional/hash encoding, increase samples/resolution, use mip-aware rendering |
| Floaters or cloudy background | Ambiguous rays, weak bounds, dynamic content, excessive splat growth | Inspect opacity/depth and splat scales | Tighten scene box, mask motion, improve coverage, prune or regularize |
| Holes in 3DGS | Sparse initialization or insufficient densification | View splat count and projected coverage | Improve SfM cloud, lower/retune densification threshold, add views |
| Exploding 3DGS memory | Over-densification or too many large projected splats | Monitor point count and radii | Raise gradient threshold, increase interval, stop densification earlier, prune, lower resolution |
| Shimmering highlights | Exposure changes, reflections, high-order angular overfit | Compare same surface across viewpoints | Stabilize lighting/exposure; compensate exposure; reduce angular capacity or use a specialized appearance model |
| Depth has no reliable scale | Monocular similarity ambiguity | Check external scale reference | Calibrate against a known distance, depth sensor, or tracked rig |

## 8. Deeper comparison: implicit field versus explicit splats

### 8.1 Optimization geometry

NeRF has a fixed-size parameter vector once its network and encoding are chosen. Capacity is distributed through learned weights and sampling decisions. Its optimization landscape is smooth in the network parameters, but each iteration can be expensive because every pixel ray invokes many field queries.

3DGS optimizes a variable-size set of primitives. Its parameters are highly interpretable—center, shape, orientation, opacity, and appearance—but structural changes are discrete: clones and splits are inserted, and primitives are pruned. The image-space gradient heuristic is effective because it directly identifies projected underfit, but it couples representation size to capture resolution, visibility, and the current state of optimization.

### 8.2 Rendering complexity

NeRF rendering cost scales with the number of rays times the number of field samples and the cost of network evaluation. Acceleration methods reduce empty-space work or replace the network with a faster encoding, but the basic ray-integration structure remains.

3DGS rendering cost scales with visible projected splats and their tile overlap. It avoids evaluating a field in empty space and maps naturally to GPU rasterization. The cost is explicit memory for each primitive and potentially many tile-splat instances when Gaussians become large on screen.

### 8.3 Editability and interoperability

Gaussians are easier to cull, transform, annotate, segment, stream, and inspect than an MLP's weights. This makes them attractive for real-time viewers and spatial applications. NeRFs are compact in some regimes and offer a continuous function that can be queried anywhere, but editing a semantic object usually requires additional fields, masks, or a conversion to another representation.

Neither representation automatically provides production-ready assets for conventional rendering. A game engine or CAD workflow may still require mesh reconstruction, UVs, material decomposition, collision geometry, or semantic segmentation.

## 9. Current research directions that matter to construction

The foundational pipelines continue to evolve in a few recurring directions:

1. **Faster field representations.** Hash grids, sparse voxel structures, proposal networks, and occupancy pruning reduce the cost of NeRF sampling.
2. **Better scale and antialiasing.** Mip-NeRF-style integrated representations reduce aliasing when the same scene is viewed at different distances or image resolutions.[^11]
3. **Unbounded-scene handling.** Contraction, proposal distillation, and distortion regularization improve outdoor and 360-degree captures.[^12]
4. **Pose-free or pose-refined reconstruction.** Jointly learning camera parameters can reduce reliance on SfM, but camera/scene ambiguity and local minima require careful initialization and regularization.[^13]
5. **Depth and geometry priors.** Monocular depth, MVS, LiDAR, or SLAM can stabilize textureless regions and improve scale, but incorrect depth priors can bias the model.
6. **Sparse-input 3DGS.** New densification and regularization methods target three-view or few-view captures, where the original ADC loop can overfit and create excessive primitives.[^19]
7. **Dynamic and 4D representations.** Time-dependent fields or moving Gaussian primitives separate scene motion from static geometry; these are necessary when the input sequence cannot be treated as a single static scene.
8. **Surface-oriented splats.** Variants constrain Gaussian geometry or use 2D/surface parameterizations to make the representation more suitable for mesh-like geometry and editing.
9. **Compression and level of detail.** Large splat scenes require pruning, quantization, hierarchy, streaming, and view-dependent level of detail to become deployable on mobile or web hardware.

These directions do not change the fundamental lesson: better representation helps only after the visual measurements are geometrically and photometrically coherent.

## 10. Conclusions

Constructing either a NeRF or a Gaussian Splat from visual data is best understood as a camera-aware inverse-rendering problem. The photographs provide color observations; SfM, calibration, SLAM, or learned pose optimization supplies the camera geometry; differentiable rendering converts a candidate scene representation back into those images; and gradient-based optimization closes the loop.

NeRF learns density and view-dependent radiance as a continuous neural function. Its core is ray construction, positional or spatial encoding, stratified and importance sampling, volume rendering, and photometric optimization. 3DGS starts with sparse geometric support from SfM, turns points into oriented translucent Gaussians, projects them to screen-space ellipses, rasterizes them with depth-aware alpha compositing, and periodically changes the number of primitives through densification and pruning.

For a new capture, the most defensible baseline is therefore: capture a static scene with overlap and parallax, estimate and validate cameras, preserve all coordinate and scale transforms, train with a held-out view split, and inspect both image quality and geometry-related diagnostics. Choose NeRF when a continuous field, mature volumetric formulation, or specialized field variant is the priority; choose 3DGS when fast training, real-time rendering, explicit primitives, and interactive inspection are the priority. In either case, the quality ceiling is set first by what the camera actually observed.

## Addendum Header

### Project direction

The practical goal is a consumer-accessible virtual-production system: inexpensive Android/IP cameras capture a moving subject and produce an engine-ready representation for compositing into a 3D environment. Required outputs are color, alpha/foreground confidence, depth, and eventually normals or material channels.

This is not ordinary static photogrammetry. It is a real-time, multi-view, deformable-subject reconstruction problem under imperfect synchronization, calibration, compression, motion blur, occlusion, and limited GPU resources.

### Recommended representation and product modes

Gaussian Splatting remains a good rendering and detail layer, but standard 3DGS is not inherently relightable. Its appearance is primarily learned from the capture lighting through view-dependent coefficients. It can composite very well, but its colors should not be assumed to behave like engine-lit albedo.

| Mode | Representation | Runtime behavior |
|---|---|---|
| **Fast** | Cached or feed-forward-predicted Gaussians plus pose deformation | Lowest latency; mostly capture-lit appearance |
| **Hybrid** | Canonical body model/SMPL-X or learned geometry plus Gaussian detail | Local RGB/depth refinement; best first target |
| **Relightable** | Canonical geometry plus Gaussian detail and albedo/normal/roughness/specular features | Slower inverse rendering; engine-light compatible |

The hybrid mode is the best first target. It provides stable motion correspondence while preserving Gaussian detail. Fast mode is a lower-latency operating point; relightable mode can follow after geometry and temporal stability are reliable.

### Neural prediction plus a persistent Gaussian cache

The proposed deep-learning direction is sound, but backpropagation should be treated as local adaptation, while the cache is an explicit persistent state. This is amortized reconstruction: a learned model predicts a plausible state quickly, and differentiable rendering reconciles it with current observations.

```text
camera streams -> synchronization/calibration -> segmentation, pose, camera tracking
               -> learned state prediction -> canonical avatar deformation
               -> persistent Gaussian/material cache -> differentiable renderer
               -> RGB/alpha/depth losses -> bounded local backpropagation -> cache update
```

The cache should be GPU-resident and spatially or body-coordinate indexed. It can contain canonical and currently deformed Gaussians, pose state, camera poses, appearance/material features, confidence, visibility history, timestamps, velocity, deformation estimates, and cache age. Each Gaussian should carry position, rotation, scale, opacity, appearance, optional material features, confidence, and visibility history.

For a moving human, avoid an unconstrained world-space cloud. Keep identity and detail in canonical coordinates, then deform them into the current pose:

\[
\mathbf{x}_t = D(\mathbf{x}_0,\mathrm{pose}_t) + R(\mathbf{x}_0,\mathrm{pose}_t),
\]

where (D) is skeletal, mesh-based, or learned deformation and (R) is a residual for clothing, hair, soft tissue, and motion not explained by the skeleton.

### Three-speed runtime design

Real-time reconstruction should be split into asynchronous loops:

- **Tracking loop (30–60 Hz):** synchronize streams, estimate cameras and pose, deform the canonical avatar, and render the current output.
- **Appearance loop (5–15 Hz):** render the cache into observed cameras and apply a few local RGB, mask, and depth gradient updates.
- **Topology/global loop (0.1–1 Hz):** correct alignment, add newly revealed regions, split/merge Gaussians, prune stale ones, and update materials.

This emulates continuous reconstruction without requiring full Gaussian optimization to converge on every frame. Display rendering remains responsive while background optimization improves the cached state.

### Online loss and update policy

An online objective can combine RGB, silhouette/alpha, depth ordering, multi-view consistency, temporal stability, deformation regularity, sparsity, and material consistency:

\[
\mathcal{L}=\lambda_{rgb}\mathcal{L}_{rgb}+\lambda_{mask}\mathcal{L}_{mask}+\lambda_{depth}\mathcal{L}_{depth}+\lambda_{mv}\mathcal{L}_{mv}+\lambda_{temp}\mathcal{L}_{temp}+\lambda_{def}\mathcal{L}_{def}+\lambda_{sparse}\mathcal{L}_{sparse}+\lambda_{mat}\mathcal{L}_{mat}.
\]

Updates should be restricted to visible, high-confidence regions. Most topology changes should remain in the slow loop. The system must distinguish pose motion, lighting change, incorrect appearance, newly visible geometry, camera drift, segmentation failure, and genuinely new geometry; otherwise backpropagation may stretch splats, corrupt colors, or create floaters.

### Implementation strategy and build order

1. **Learned initializer:** predict pose, depth, geometry, and initial Gaussians, then perform a short local optimization. This is the safest first prototype.
2. **Learned updater:** provide the previous cache and new multi-view observations to a causal/recurrent network that predicts updates.
3. **Learned renderer plus explicit residual cache:** let the network predict common human motion/appearance while the cache preserves subject-specific identity and detail.

Build in this order: calibrated static capture and engine export; canonical body plus Gaussian detail; learned initialization and local RGB/mask/depth refinement; asynchronous densification/pruning and recovery; then normals/material features for relighting.

Existing research can supply pose tracking, segmentation, depth priors, multi-view geometry, differentiable Gaussian rendering, streaming splatting, animatable Gaussian avatars, and relightable-avatar components. The project-specific research contribution is more likely to be the deformable Gaussian neural cache: correspondence, confidence-aware reads/writes, temporal stabilization, asynchronous topology maintenance, recovery from tracking failure, and reliable engine export.

The central hypothesis is: **a learned predictor supplies a plausible subject state immediately; a persistent canonical Gaussian cache preserves identity and detail; differentiable rendering supplies evidence; and bounded asynchronous backpropagation gradually reconciles the cache with live observations.**

## Sources

[^1]: COLMAP Team, “Tutorial — Structure-from-Motion and Multi-View Stereo,” COLMAP documentation. [https://colmap.github.io/tutorial](https://colmap.github.io/tutorial)

[^2]: COLMAP Team, “Tutorial — Structure-from-Motion and Multi-View Stereo,” capture guidance and pipeline description. [https://colmap.github.io/tutorial](https://colmap.github.io/tutorial)

[^3]: Johannes L. Schönberger and Jan-Michael Frahm, “Structure-from-Motion Revisited,” CVPR 2016. [PDF](https://www.cv-foundation.org/openaccess/content_cvpr_2016/app/S18-10.pdf)

[^4]: Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng, “NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis,” ECCV 2020. [arXiv](https://arxiv.org/abs/2003.08934); official implementation: [https://github.com/bmild/nerf](https://github.com/bmild/nerf)

[^5]: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis, “3D Gaussian Splatting for Real-Time Radiance Field Rendering,” ACM Transactions on Graphics 42(4), 2023. [arXiv](https://arxiv.org/abs/2308.04079); project page: [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

[^6]: Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, and Victor Adrian Prisacariu, “NeRF--: Neural Radiance Fields Without Known Camera Parameters,” 2021. [arXiv](https://arxiv.org/abs/2102.07064)

[^7]: GraphDeco / Inria, “gaussian-splatting,” official reference implementation and training documentation. [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

[^8]: COLMAP Team, “Database Format” and “Output Format,” COLMAP documentation. [Database Format](https://colmap.github.io/database.html); [Output Format](https://colmap.github.io/format.html)

[^9]: Nerfstudio Team, “Using Custom Data,” documentation. [https://docs.nerf.studio/quickstart/custom_dataset.html](https://docs.nerf.studio/quickstart/custom_dataset.html)

[^10]: Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller, “Instant Neural Graphics Primitives with a Multiresolution Hash Encoding,” ACM Transactions on Graphics 41(4), 2022. [arXiv](https://arxiv.org/abs/2201.05989)

[^11]: Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan, “Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields,” ICCV 2021. [arXiv](https://arxiv.org/abs/2103.13415)

[^12]: Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman, “Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields,” CVPR 2022. [arXiv](https://arxiv.org/abs/2111.12077)

[^13]: Wang et al., “NeRF--,” especially Sections 4–5 on joint camera/field optimization and its ambiguities. [https://arxiv.org/abs/2102.07064](https://arxiv.org/abs/2102.07064)

[^14]: GraphDeco / Inria, `dataset_readers.py`, synthetic-data initialization path. [https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/dataset_readers.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/dataset_readers.py)

[^15]: GraphDeco / Inria, `scene/gaussian_model.py`, covariance parameterization and `create_from_pcd` initialization. [https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py)

[^16]: GraphDeco / Inria, `train.py`, progressive SH-degree activation and photometric loss. [https://github.com/graphdeco-inria/gaussian-splatting/blob/main/train.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/train.py)

[^17]: GraphDeco / Inria, `rasterizer_impl.cu`, tile/depth key generation and GPU rasterization. [https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/rasterizer_impl.cu](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/rasterizer_impl.cu)

[^18]: GraphDeco / Inria, `scene/gaussian_model.py`, gradient accumulation, clone/split/prune logic. [https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py)

[^19]: Samuel Rota Bulò, Lorenzo Porzi, and Peter Kontschieder, “Revising Densification in Gaussian Splatting,” 2024. [arXiv](https://arxiv.org/abs/2404.06109); Glenn Grubert, Florian Barthel, Anna Hilsmann, and Peter Eisert, “Improving Adaptive Density Control for 3D Gaussian Splatting,” 2025. [arXiv](https://arxiv.org/abs/2503.14274)

[^20]: Nerfstudio Team, “NeRF,” assumptions and pipeline documentation. [https://docs.nerf.studio/nerfology/methods/nerf.html](https://docs.nerf.studio/nerfology/methods/nerf.html); camera conventions: [https://docs.nerf.studio/quickstart/data_conventions.html](https://docs.nerf.studio/quickstart/data_conventions.html)
