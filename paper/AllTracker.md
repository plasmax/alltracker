Sections:
Abstract
1 Introduction
2 Related Work
    Optical flow
    Flow-based point tracking
    Point trackers without flow
    Training data and self-supervision
3 AllTracker
    3.1 Encoding
    3.2 Computing appearance similarity
    3.3 Track initialization
    3.4 Iterative refinement
    3.5 Model training
    3.6 Implementation details
4 Experiments
    Metrics and benchmarks
    Baselines
    AllTracker variants
    4.1 Main results
        High-resolution performance
        Speed
        Realtime inference
        Additional metrics
        Performance on flow benchmarks
        Qualitative results
    4.2 Ablations
        Temporal module
        Motion representation
        Backbone
        Hyperparameters
        Huber vs. L1
        Does frame ordering matter?
5 Conclusion and Limitations
    Acknowledgments
References
Appendix A Additional model details
    Recurrent module
    Model layers
    AllTracker-Tiny
    Initialization strategy
Appendix B Additional training details
Appendix C Additional baseline details
Appendix D Additional optical flow results
Appendix E Additional ablation details
    Validation dataset
    Inference steps
Appendix F Additional qualitative results

Files Content:

## Contents
- 1 Introduction
- 2 Related Work
  - Optical flow
  - Flow-based point tracking
  - Point trackers without flow
  - Training data and self-supervision
- 3 AllTracker
  - 3.1 Encoding
  - 3.2 Computing appearance similarity
  - 3.3 Track initialization
  - 3.4 Iterative refinement
  - 3.5 Model training
  - 3.6 Implementation details
- 4 Experiments
  - Metrics and benchmarks
  - Baselines
  - AllTracker variants
  - 4.1 Main results
    - High-resolution performance
    - Speed
    - Realtime inference
    - Additional metrics
    - Performance on flow benchmarks
    - Qualitative results
  - 4.2 Ablations
    - Temporal module
    - Motion representation
    - Backbone
    - Hyperparameters
    - Huber vs. L1
    - Does frame ordering matter?
- 5 Conclusion and Limitations
  - Acknowledgments
- References
- Appendix A Additional model details
  - Recurrent module
  - Model layers
  - AllTracker-Tiny
  - Initialization strategy
- Appendix B Additional training details
- Appendix C Additional baseline details
- Appendix D Additional optical flow results
- Appendix E Additional ablation details
  - Validation dataset
  - Inference steps
- Appendix F Additional qualitative results

## Abstract

Abstract We introduce AllTracker: a model that estimates long-range point tracks by way of estimating the flow field between a query frame and every other frame of a video. Unlike existing point tracking methods, our approach delivers high-resolution and dense (all-pixel) correspondence fields, which can be visualized as flow maps. Unlike existing optical flow methods, our approach corresponds one frame to hundreds of subsequent frames, rather than just the next frame. We develop a new architecture for this task, blending techniques from existing work in optical flow and point tracking: the model performs iterative inference on low-resolution grids of correspondence estimates, propagating information spatially via 2D convolution layers, and propagating information temporally via pixel-aligned attention layers. The model is fast and parameter-efficient (16 million parameters), and delivers state-of-the-art point tracking accuracy at high resolution (i.e., tracking 768 × 1024 768\times 1024 768 × 1024 pixels, on a 40G GPU). A benefit of our design is that we can train jointly on optical flow datasets and point tracking datasets, and we find that doing so is crucial for top performance. We provide an extensive ablation study on our architecture details and training recipe, making it clear which details matter most. Our code and model weights are available: https://alltracker.github.io

## 1 Introduction

Estimating the long-range trajectories of arbitrary points in the world, using sequences of 2D images as input, has been a concrete and competitive challenge in computer vision since at least 2006 [40].
The utility of optical flow (i.e., the instantaneous velocity of pixels [16]) toward this goal has long been obvious, yet it has remained challenging to upgrade flows into long-range tracks.

Instantaneous flows can be “chained” into multi-frame tracks, by interpolating in one flow field at the endpoints of the previous flow field, but imperfect flows will accumulate drift, and such chains must also be carefully stopped at occlusions [40].
An attractive shortcut here is to directly compute the flow between a reference frame and each other frame, and thus track drift-free and across occlusions, but estimating “long-range flow” becomes increasingly difficult as the time interval widens between the reference frame and the target, due to increasing variation in perspective, illumination, and scene geometry.

In light of these challenges, recent work has side-stepped optical flow and created
a line of bespoke point trackers focused on learning multi-frame temporal priors for
reducing drift and tracking through occlusions [19, 59, 25, 12, 24]. These works explicitly highlight that the limited temporal context in flow-based methods is a severe weakness, and have made substantial progress by addressing this. However, the proposed solutions add temporal awareness at the cost of spatial awareness, and only deliver tracks for sparse sets of points. Recent attempts at “dense” point tracking [27, 50, 37] are not as accurate as the latest sparse trackers, and struggle with high-resolution input.

In this paper, we demonstrate that learnable multi-frame temporal priors can be built together with high-resolution spatial awareness,
by casting point tracking as a multi-frame long-range optical flow problem, as illustrated in Figure [1](https://arxiv.org/html/2506.07310v2#S0.F1).

Our design combines ideas from point tracking literature and optical flow literature, arriving at a composite architecture with broader capability. Following the trend in both areas, the heart of the model is a recurrent module that iteratively improves motion estimates [47, 19]. This module relies on information from spatial cross-correlations, giving it a strong inductive bias for feature-matching, which speeds up training.
From point tracking literature, we borrow a per-pixel temporal module, to learn a motion prior and track through occlusions [19].
From optical flow literature, we borrow the idea of performing the majority of processing on a low-resolution grid, to allow fast spatial message-passing via 2D convolutions, and recover spatial precision at the end of the architecture with an upsampling layer [47, 54].
In relation to other point trackers, the key novelty of our approach is that we frame the problem as long-range dense flow, which makes sparse methods (of similar speed and accuracy) redundant.
In relation to other optical flow models, the key novelty of our approach is that we solve a window of flow problems simultaneously, instead of frame-by-frame; the information shared within and across windows unlocks the capability to resolve flows across wide time intervals.

A secondary benefit of our design is that we can train jointly on optical flow datasets and point tracking datasets. We therefore build a mix of publicly-available datasets to support our model, spanning different time-ranges, annotation densities, and resolutions, and train with uniform sampling across this mix. We find that this strategy, paired with a long training schedule, is crucial for top performance.

In summary, our method blends techniques and datasets from optical flow literature and point tracking literature, and the resulting model is a state-of-the-art point tracker that operates in high resolution at full density. Inspired by CoTracker [25] and highlighting our capability for all-pixel tracking, we name our approach AllTracker. We will release our model, our data, and our code.

## 2 Related Work

#### Optical flow

The introduction of the concept of optic flow can attributed to Gibson [16] (see Niehorster’s review [38]). In computer vision, optical flow methods generally take two consecutive frames of video as input, and estimate the motion that relates the pixels of the first frame to the pixels of the second frame [21, 32].
The dominant technique, from classic approaches to current learning-based approaches, is to iterate on a solution: first, candidate correspondences are initialized (e.g., with zero-motion or constant velocity), then the “costs” of these correspondences are computed by measuring how well the appearance features match, and this appearance cost is combined with smoothness terms or learned priors, and then better correspondences are sought in the neighborhood of the current solution [4, 22, 58, 42, 47]. Our work uses the same basic technique.

The current state of the art in optical flow estimation is SEA-RAFT [54].
SEA-RAFT first directly estimates a low-resolution flow field [14, 22], then computes appearance features with a ResNet-34 [20], and computes pixel-to-pixel correlations with these features [58], then iteratively estimates refinements to the flow via 2D convolutions that take flows and correlations as input, and finally upsamples the flow to full resolution via a pixel-shuffle layer [41].
Our work takes inspiration from SEA-RAFT (and its predecessor RAFT [47]) in the way that we iteratively refine low-resolution flows and upsample to full resolution, but we operate on multiple frames at a time (i.e., 16 frames instead of 2), and we add temporal attention layers to help information propagate across the time axis.

#### Flow-based point tracking

Optical flow is often used as a building block toward multi-frame tracking. The standard technique in this space is to “chain” flow vectors end-to-end to form longer tracks, and guard against drift by monitoring for occlusions [40, 45]. Recent techniques skip past occlusions by using multistep flows [9, 7, 8, 36], or attempt to track through occlusions using a learned temporal prior [5] (inspired by non-flow point trackers). These methods deliver dense multi-frame trajectories similar to our method, but first require an expensive pre-processing stage in which optical flows are estimated by a different model, while our method handles the full problem efficiently with a single model.

We highlight two concurrent works which propose a flow-based method somewhat similar to our own: DTF [50] and DELTA [37].
Both iteratively estimate flows to relate a reference frame to other frames,
but unlike our work, these papers propose special-purpose transformer-based methods in which global spatial message-passing is approximated by cross-attending to sparse “anchor” or “centroid” tokens. Our method simply uses 2D convolutions for this step, more like SEA-RAFT [54].
We note also that DTF and DELTA both struggle with high-resolution inputs (running out of memory on our 40G GPUs), despite DELTA using low-resolution inference and high-resolution upsampling, comparable to SEA-RAFT and our approach.
In performance, DTF is not competitive with recent point trackers (while ours exceeds them); DELTA’s main model requires depth input, but we compare against its 2D variant and outperform it.

#### Point trackers without flow

Many classic methods track points directly, without relying on optical flow as a submodule [32, 49].
Harley et al. [19] recently introduced a new method in this space,
with some components similar to flow models [47], but tracking points independently, and adding a multi-frame inference window which enabled the model to track through occlusions. That work, along with a new “Tracking Any Point” benchmark that appeared soon afterward [11], spurred much follow-up effort, involving multi-point context [25, 24], better initialization schemes [12], wider correlation context [2, 6], new transformer-based designs [30, 29], and post-hoc densification methods [27].

The current state of the art in point tracking is CoTracker3 [24]. CoTracker3 first computes feature maps for all frames, then initializes tracks for a sparse set of query points, then retrieves local correlations based on these tracks, and interleaves modules which pass information within tracks (temporally) and across tracks (spatially). The main novelty is in the spatial propagation: each point attends to a set of latent “virtual points” which learn a compressed representation of the video motions.
While CoTracker3 delivers very accurate tracks, it requires users to pre-select a sparse set of points to track, and results vary based on how the queries are distributed [25]. CoTracker3 is more efficient than its predecessors [25, 19], but is still limited to tracking a few thousand points at a time.
Our approach is simpler and more memory-efficient: we track on a low-resolution grid, where we achieve message-passing with simple 2D convolutions, and we recover spatial precision at the end of the architecture with a fast upsampling layer [41].
We note that this efficient upsampling technique was known well before the recent resurgence of point trackers [41], but perhaps its relevance was not fully appreciated.

#### Training data and self-supervision

Current methods in optical flow and point tracking are data-driven, making data and supervision choices crucial. In optical flow, state-of-the-art methods rely on a combination of mostly synthetic datasets,
including FlyingChairs [14], FlyingThings3D [34], Monkaa [34], Driving [34], AutoFlow [43], Spring [35], VIPER [39], HD1K [26], KITTI [15], and TartanAir [53]. In point tracking, the main training datasets are Kubric [17], FlyingThings++ [19], and PointOdyssey [59].
There is an art to creating a curriculum from diverse datasets
to optimize downstream performance on particular benchmarks [54], but in our work we simply concatenate the datasets and shuffle the samples. We note that all previous point tracking methods did not use flow data, perhaps because most models do not support 2-frame inference or dense output; we therefore also provide experiments using Kubric alone.

Motivated by the fact that the training data for point trackers is all synthetic, several groups have explored self-supervision methods based on boot-strapping pre-trained models using pseudo-labels computed on real videos [44, 13, 24].
However, these schemes offer only minor gains over the supervised weights where bootstrapping begins. In this work, we avoid any pseudo-labelling and simply add more synthetic data, capitalizing on our model’s ability to accept supervision from optical flow data.

## 3 AllTracker

Figure: Figure 2: AllTracker architecture. First, we compute feature maps for all frames, and copy the zeroth (query) feature map to every timestep, and compute multi-scale cost volumes. Then, we iterate a recurrent module, which references the query feature map and cost volume pyramid at each timestep, and estimates a low-resolution correspondence field, using interleaved 2D convolutions and pixel-aligned temporal attentions. The output of the RNN is upsampled into high-resolution optical flow maps, which relate all pixels of the zeroth frame to every other frame.
Refer to caption: x2.png

Our model takes a video as input, along with a “query” index, specifying which frame’s pixels to track, and it outputs full-timespan tracking for all of the pixels in that frame.

Concretely, we are given as input a video of shape $T,H,W,3$, where $T$ is the number of frames, and $H,W$ indicate the spatial resolution of each frame. We are also given an index $t\in T$,
indicating which frame has the pixels we need to track. Our final output is a tensor shaped $T,H,W,4$: the first two channels are optical flow maps indicating the offset that takes each pixel in the query frame to its correspondence in every other frame, and the next two channels estimate visibility and confidence.

A brief summary of the method is as follows. We operate in sliding window fashion across $T$, processing subsequences of length $S$, advancing at a stride of $S/2$.
We begin by quickly computing low-resolution feature maps for the subsequence, yielding a feature volume with dimensions $S,H/8,W/8,D$, where $D$ is the channel dimension of the features.
We then initialize a low-resolution tensor of outputs, shaped $S,H/8,W/8,4$, and iteratively revise this tensor
using cross correlations and features across space and time as reference.
Finally, we upsample the outputs, producing full resolution optical flow, visibility, and confidence, shaped $S,H,W,4$.
We then advance our window by $S/2$, using previous estimates as initialization,
and repeat inference.
Figure [2](https://arxiv.org/html/2506.07310v2#S3.F2) shows the method in a diagram.

### 3.1 Encoding

Our first stage encodes the input video frames into low-resolution feature maps. We achieve this with a ConvNeXt-Tiny [31] model (see Sec. [3.6](https://arxiv.org/html/2506.07310v2#S3.SS6) for implementation details).
This encoder compresses a subsequence input of shape $S,H,W,3$ to the shape $S,H/8,W/8,D$, where $D$ is the embedding dimension (in our case $D=256$), $S$ is the subsequence window length (in our case $S=16$). When the full video length $T$ is not too large (e.g., at training time), we compute all of the feature maps in parallel.

We then isolate the feature map of the query frame and tile it to the length of the subsequence, so that we have a copy for each timestep, as illustrated in Figure [2](https://arxiv.org/html/2506.07310v2#S3.F2).

### 3.2 Computing appearance similarity

Using the feature maps, we build a multi-scale 4D correlation volume, capturing appearance-based tracking cues.
We implement this in two steps: first we convert each feature map into a feature pyramid, by average-pooling at multiple strides (in our case 5 strides: $\{1,2,4,8,16\}$), and then we cross-correlate the query feature map with these pyramids. That is, for each feature vector in the query feature map, we compute its dot product across each timestep’s pyramid.

The output of this step can be understood as a large collection of heatmaps: one heatmap per tracked pixel, per scale, per timestep.
The model will index into these heatmaps to retrieve
information on where correspondences lie. When a target’s correspondence is visible at a certain timestep, we expect the respective heatmap to have a strong peak at the location of the correspondence.

### 3.3 Track initialization

We next initialize estimates for tracking coordinates, visibility, and confidence, by creating a tensor with shape $S,H/8,W/8,4$, where the coordinate channels are initialized with a 2D meshgrid, and visibility and confidence are initialized with zeros. If we are on a window beyond the first, we populate this tensor with the estimates that overlap, and copy forward the values from the last estimated timestep.

### 3.4 Iterative refinement

The most important stage of our model is the iterative refinement stage, where we update all of the estimates. We describe the main points of the module here, but also provide extensive details (and a diagram) in the supplementary.

For each timestep, for each pixel in the query frame, we have: a feature vector $\mathbf{f}$ (length $D$), visibility and confidence estimates $\mathbf{v,c}$ (together length 2), a position estimate $\mathbf{p}$ (length $2$), and a correlation pyramid $\{\mathbf{C}_{1},\mathbf{C}_{2},\ldots\}$ shaped $\{H/8\times W/8,H/16\times W/16,\ldots\}$.
We convert this data into a “local” representation suitable for convolutions, as follows.
For the position data, we replace the absolute position estimates $\mathbf{p}$ with motion estimates, by subtracting the source positions: $\mathbf{m}=\mathbf{p}-\mathbf{p}_{0}$.
For each level of the correlation pyramid, we extract a small patch centered at $\mathbf{p}$, and flatten these patches into a single vector $\mathbf{q}$ of length $L\cdot(2R+1)^{2}$, where $R$ is a radius parameter for the patches (e.g., 4) and $L$ is the number of levels in the pyramid (e.g., 5). All together, we have
$\mathbf{f},\mathbf{v},\mathbf{c},\mathbf{m},\mathbf{q}$ (per pixel, per timestep), which in our setup makes $665$ channels.

Inside the recurrent module, we process the input data using interleaved spatial and temporal blocks. Each spatial block is a 2D ConvNeXt block; each temporal block is a pixel-aligned transformer block. By pixel-aligned, we mean that attention only happens along the temporal axis (i.e., with cost quadratic in $S=16$), and this is done for every pixel in parallel.
We note that an important convenience of our design is that all of the tensors in this stage are aligned with the “query” frame; therefore pixel-aligned attention is attention between corresponding pixels. After the interleaved spatial and temporal blocks propagate tracking-related information across our window, we decode explicit revisions for visibility, confidence, and motion.

We apply visibility, confidence, and motion revisions via simple summation with the previous values
$\mathbf{x}_{\textrm{new}}=\mathbf{x}_{\textrm{old}}+\delta\mathbf{x}$ for $\mathbf{x}\in\{\mathbf{v},\mathbf{c},\mathbf{m}\}$,
where $\delta\mathbf{x}$ denotes an explicit revision produced by the model.

We additionally decode weights for a pixel-shuffle upsampling step [41, 47]. We apply this upsampling to our visibility, confidence, and motion maps, bringing them from $1/8$ resolution to full resolution.

We iterate our recurrent “refinement” stage 4 times, sharing weights. At training time, we use the outputs from every refinement step for supervision, and at test time we only use the final iteration’s output.

### 3.5 Model training

In the datasets which we use for training, we have supervision in the form of either optical flow or sparse point tracks, which we simply treat as trajectories of varying lengths. We use an $L_{1}$ loss between estimated trajectories and ground truth, with a higher weight on loss for visible points:

$$ $L_{\textrm{track}}=\alpha\sum_{k}^{K}\gamma^{K-k}(\mathbbm{1}_{\textrm{occ}}/5+\mathbbm{1}_{\textrm{vis}})||P_{k}-\hat{P}||_{1},$ (1) $$

where $P_{k}$ is a set of estimated trajectories at refinement step $k$, $\hat{P}$ is the corresponding ground truth, $\gamma$ (set to $0.8$) makes later refinement steps weigh more in the loss, and $\alpha$ is a balancing hyperparameter (set to $0.05$).

We supervise our visibility and confidence maps with a binary cross entropy loss. We ask the visibility estimates to match ground truth binary labels: $L_{\textrm{vis}}=\sum_{k}^{K}\textrm{BCE}(V,\hat{V})$.
We ask the confidence estimates to reflect whether or not the corresponding position estimates are within $12$ pixels of ground truth [12, 24]:
$L_{\textrm{conf}}=\sum_{k}^{K}\textrm{BCE}(C,\mathbbm{1}[||X_{k}-\hat{X}||_{2}<12])$.
We apply these losses at every refinement step.

To supervise the model with sparse annotations, we use the coordinates of the ground truth to sample our corresponding estimates,
and apply the loss at these sparse locations.
We use bilinear sampling for trajectories, and nearest-neighbor sampling for visibility and confidence.

### 3.6 Implementation details

We provide extensive implementation details in the supplementary material, but describe the main details here.

Our CNN backbone is based on a pre-trained ConvNeXt-Tiny [31]. We use the first three “blocks” of this architecture, totaling 12.72 million parameters. We convert the third block from stride 2 to stride 1, by applying bicubic interpolation to the $2\times 2$ stride-$2$ kernel to create a $3\times 3$ stride-1 kernel, and re-scale these weights according to the change in area (i.e., scaling by $4/9$).

Our full model is 16.48 million parameters. We train it using 8 A100-40G GPUs. We train in two stages: first on Kubric [17] for 200,000 steps at a learning rate of $5e-4$, and then on a mix of point tracking and optical flow datasets for 400,000 iterations at a learning rate of $1e-5$. We use the standard augmentations from prior work in point tracking [19] and optical flow [47], which consist of random shifting and scaling, color jitter, and square occlusions.

Our optical flow datasets include FlyingChairs [14], FlyingThings3D [34], Monkaa [34], Driving [34], AutoFlow [43], Spring [35], VIPER [39], HD1K [26], KITTI [15], and TartanAir [53]. Our point tracking datasets include Kubric [17], DynamicReplica [23],
and PointOdyssey [59]. We also follow Harley et al. [19] and compute point tracks from optical flows where possible, making FlyingThings++, Monkaa++, Driving++, and Spring++.

For trajectory estimates, we apply the model’s additive revisions directly in pixel coordinates. We also supervise in pixel coordinates, which is a reason to scale the loss by the factor $\alpha=0.05$.
For visibility and confidence estimates,
we apply the model’s revisions on logits (rather than on probabilities),
but we apply a sigmoid to these values before they are input to the refinement block, to stabilize their range.

Figure: Table 1: Comparison against recent point trackers and optical flow models, across nine datasets. We evaluate $\delta_{\text{avg}}$ (higher is better), using an input resolution of $384\times 512$. The benchmarks are BADJA [3], CroHD [46], TAPVid-DAVIS [11], DriveTrack [1], EgoPoints [10], Horse10 [33], TAPVid-Kinetics [11], RGB-Stacking [28], and RoboTAP [52].

Figure: Table 2: High-resolution comparison between our model and CoTracker3 [24], which is the previous state of the art. We evaluate $\delta_{\text{avg}}$ (higher is better) on nine point tracking benchmarks, at resolutions $448\times 768$ and $768\times 1024$. Parameter counts are in millions.

## 4 Experiments

This section summarizes the key results, which concern the accuracy of our tracker compared to prior state of the art, on a diverse array of test benchmarks. We provide additional details for our experiments in the appendices.

#### Metrics and benchmarks

Our main metric is $\delta_{\textrm{avg}}$: an accuracy metric with max value 100, capturing how closely the estimated trajectory positions follow the ground truth positions. This is defined as the average of multiple $\delta_{k}$ metrics, where $\delta_{k}$ equals $100$ when the estimate is within $k$ pixels of ground truth, measuring at $256\times 256$ resolution:

$$ $\delta_{k}=100\cdot\mathbbm{1}[||\mathbf{p}-\mathbf{\hat{p}}||_{2}<k].$ (2) $$

Averaging $\delta_{k}$ over $k\in\{1,2,4,8,16\}$ yields $\delta_{\textrm{avg}}$. We prioritize this over other considered metrics (e.g., L2 [19], PCK [19], AJ [11]) because we find it is both interpretable and robust to outliers. In certain benchmarks we also compute occlusion classification accuracy, and Average Jaccard (AJ) which mixes tracking accuracy with occlusion accuracy.

We evaluate on a total of nine publicly-available benchmarks with point annotations, covering a wide set of domains that include animals (BADJA [3], Horse10 [33]), YouTube videos (TAPVid-DAVIS [11], TAPVid-Kinetics [11]), surveillance camera recordings (CroHD [46]), egocentric recordings (EgoPoints [10]), and robotics data (RGB-Stacking [28], RoboTAP [52]). We trim the video lengths to a maximum of 600 frames, and on the larger datasets (CroHD, DriveTrack, EgoPoints, Horse10, Kinetics, RoboTAP) we only track points from the first available query frame, which we find gives similar results to using all possible queries and full video lengths. We argue that using a wide array of benchmarks helps flatten the effects of label noise, and gives a better sense of the overall performance in practice. We encourage future work to follow our example, but we note that our full evaluation is time-consuming (with some baselines requiring multiple days to finish their pass).

#### Baselines

We benchmark a variety of recent point trackers and optical flow models in our tests, including the optical flow models RAFT [47] and SEA-RAFT [54], the long-term flow method AccFlow [55], and point trackers PIPs++ [59], LocoTrack [6], DELTA [37], BootsTAPIR [13]. We make close comparisons against DELTA [37] and CoTracker3 [24], which are the most recent state-of-the-art point trackers.
DELTA is relevant because it is a concurrent work focused on dense tracking.
CoTracker3 is relevant because it outperforms all past work. It has two variants: one trained exclusively on Kubric, and another version finetuned on pseudolabels produced by an ensemble of point trackers on a curated set of 15,000 real videos. This finetuned version of CoTracker3 outperforms all past work.(^1^11The CoTracker3 repo also includes an “offline” variant, but it is not consistently better than the standard version, and in any case its memory consumption is outside our compute budget.) We note that the performance of CoTracker3 varies depending on how the query points are grouped [25]. The paper suggests to run each query in a separate forward pass, while also adding “support” points around the query and a sparse grid covering the image, but in our multi-benchmark evaluation this would be prohibitively expensive (e.g., weeks). We simply give CoTracker3 all queries at once, supplemented by a sparse grid of points around the image. Comparing with values reported by the authors (reproduced in the supplementary), we find that our setup over-estimates CoTracker3’s accuracy, but makes our evaluation tractable.

#### AllTracker variants

In addition to our main model, which is 16.48 million parameters and trained on a mix of datasets, we evaluate a “tiny” version, which is trained similarly to the main model but uses a cheaper CNN backbone and totals only 6.29 million parameters (see details in supplemental). For both versions, we additionally report the results with models trained exclusively on Kubric.

### 4.1 Main results

We evaluate our method’s ability to track arbitrary points in diverse videos of various lengths, and compare against the state of the art in Table [1](https://arxiv.org/html/2506.07310v2#S3.T1). For this evaluation we resize all input videos to $384\times 512$. We note that DELTA (the only other dense point tracker) often runs out of memory when testing on a 40G GPU, so we use a 96G GPU to evaluate it.

As shown in the table, even our weakest variant, AllTracker-Tiny-Kub, is competitive with state of the art on many datasets, despite being only 6.29M parameters and training on a single synthetic dataset. Our full AllTracker model outperforms all other models on average, with the closest competitor being
the variant of CoTracker3 which relied on an expensive bootstrapping scheme (noted by “+15k” in the table).
This CoTracker3 model wins on three datasets (CroHD, Davis, DriveTrack) and our model wins on the rest and wins on average (66.1 vs. 65.0). Our method has a substantial gain over past work on the RGB-Stacking benchmark, which has many query points inside textureless regions; our model’s good performance here suggests that it is able to incorporate spatial context from a wider area than the previous models.
We note that CoTracker3’s bootstrapping technique could be combined with our contributions, but it is simpler to gather additional synthetic data as we have done.

Comparing models trained on Kubric to models trained with a broader data distribution, we see that the additional training data does not reliably improve performance in the surveillance dataset CroHD [46] or the driving dataset DriveTrack [1]. This suggests room for improvement, in terms of perhaps data balancing or model capacity.

#### High-resolution performance

Focusing on the comparison between our model and CoTracker3, we evaluate these models at higher resolutions in Table [2](https://arxiv.org/html/2506.07310v2#S3.T2). Here we see that our model’s performance reliably increases at higher resolutions, while CoTracker3 plateaus after $448\times 768$, leading to the surprising result that AllTracker-Tiny outperforms CoTracker3 at $768\times 1024$.
Our model is also more memory efficient: with AllTracker we can process videos at $768\times 1024$ (and produce 786,432 tracks at once) on a 40G A100 GPU, while with CoTracker3, despite only tracking sparse points, we encountered out-of-memory errors until eventually performing the tests on a 96G H100 GPU. Our model’s relative memory efficiency is partly from using a spatial stride of 8 in the encoder, while CoTracker3 uses stride 4.

#### Speed

We compare the throughput of our model to other point trackers and optical flow models in Figure [3](https://arxiv.org/html/2506.07310v2#S4.F3): our model runs at approximately the speed of optical flow methods, while achieving accuracies higher than the point trackers.

Figure: Figure 3: AllTracker (top right corner) delivers accurate multi-frame tracks at the throughput of an optical flow model.
Refer to caption: x3.png

Figure: Table 3: Average Jaccard evaluation on TAP-Vid datasets.

Figure: Table 4: Occlusion accuracy evaluation on TAP-Vid datasets.

#### Realtime inference

Our model’s sliding-window strategy can be run in “streaming” fashion real-time, at a penalty to accuracy: at $512\times 512$ our model runs at 57.9 FPS and achieves 62.6 $\delta_{\text{avg}}$ (vs. 66.1 normally); BootsTAPIR [13] has also published a realtime variant, which runs at 21.4 FPS and reaches 62.2 $\delta_{\text{avg}}$.

Figure: Figure 4: AllTracker produces accurate displacement fields across dozens of frames. Prior optical flow methods struggle to make correspondences across wide time gaps, while our model uses temporal priors to resolve the ambiguity; prior point trackers take multiple minutes to produce output at this density, and show splotchy pattern errors, while our method produces coherent output in less than a second.
Refer to caption: x4.png

Figure: Table 5: A transformer-based temporal module performs better than mixer-based or convolution-based variants.

Figure: Table 7: Hyperparameter search: It is best to use a ConvNeXt [31] backbone, 3 refinement blocks, and radius-4 correlations at 5 scales.

#### Additional metrics

Evaluating AJ and occlusion accuracy on TAPVid datasets, we find that AllTracker obtains substantially better AJ than CoTracker3: 68.9 vs. 63.1 on average (see Table [3](https://arxiv.org/html/2506.07310v2#S4.T3), and slightly better occlusion accuracy: 91.5 vs. 89.3 (see Table [4](https://arxiv.org/html/2506.07310v2#S4.T4)). We perform this evaluation only on the TAP-Vid datasets because we find that other datasets’ visibility labels are not as reliable.

#### Performance on flow benchmarks

While traditional optical flow is not our focus, it is interesting to inspect the model’s accuracy in this task. We submitted AllTracker to the official SINTEL test benchmark, yielding end-point error scores of 1.673 on “clean” and 3.244 on “final”; this is not as good as SEA-RAFT [54] (1.309 / 2.601) but comparable to GMFlow [57] (1.736 / 2.902). We note that it is common in optical flow literature to produce a different model for each benchmark (finetuned with a particular data mix and resolution), but for this test we simply use our main model as-is. Qualitatively, AllTracker’s flow maps appear to be coarser than the ones from SEA-RAFT [54], suggesting that the model is underfitting on this task.
We provide more optical flow results in the supplementary.

#### Qualitative results

We visualize the long-range flow maps for our model and selected baselines in Figure [4](https://arxiv.org/html/2506.07310v2#S4.F4). Note that computing the dense flow map for the sparse methods takes multiple minutes, while our method (and the flow methods) produce it in under a second. We find that these flow visualizations reveal differences in the spatial coherence of the tracks: sparse point trackers from the past few years (PIPs++, LocoTrack, CoTracker3) show progressive improvement but include splotchy pattern errors in the motion fields. The optical flow methods (AccFlow, RAFT, SEA-RAFT) show better spatial coherence, but the estimates are unreliable when the displacements are too large. Our method produces motion fields that are both coherent and accurate.

### 4.2 Ablations

We verify our design choices in an exhaustive series of ablation studies.
In these experiments, we train each model for 100,000 iterations with 2 GPUs, using Kubric, with 4 inference iterations per sample, at a learning rate of 4e-4, with videos of size $24\times 384\times 512$ and $56\times 256\times 256$. We test at resolution $384\times 512$ on a validation dataset that we create from BADJA [3], CroHD [46], TAPVid-DAVIS [11], DriveTrack [1], Horse10 [33], and RoboTAP [52], and report the mean $\delta_{\textrm{avg}}$ across this set. We find that this truncated training and evaluation setup is crucial for enabling a thorough exploration of the model space. We note that the absolute values of these ablation experiments should not be compared to the main experiments, but can be compared to each other.

#### Temporal module

Prior work has explored different network architectures for learning tempral priors: PIPs [19] used an MLP-Mixer [48], PIPs++ [59] and TAPIR [12] used 1D convolutions (which might reasonably be upgraded to 1D ConvNeXt layers [31]), and CoTracker [25] used a transformer [51].
In Table [6](https://arxiv.org/html/2506.07310v2#S4.T6) we compare these in our setup and find that a transformer works best. We additionally note that the transformer option is most amenable to changing the window size, which is important for jointly training on optical flow (e.g., convolution kernels of size 3 cannot apply to an input of length 2).

#### Motion representation

Most point trackers use sinusoidal position embeddings of motion, and methods vary on whether displacements should be relative to the query frame or to an adjacent frame.
Table [6](https://arxiv.org/html/2506.07310v2#S4.T6) shows that displacement relative to the query frame (i.e., frame 0) works best. This choice has the additional benefit of merging the problem with long-range optical flow. Table [6](https://arxiv.org/html/2506.07310v2#S4.T6) also shows that sinusoidal embeddings do not help.

#### Backbone

Most point trackers use a “BasicEncoder” backbone which originated from RAFT [47]. We show in Table [7](https://arxiv.org/html/2506.07310v2#S4.T7) that a pre-trained ConvNeXt backbone performs better, while a ConvNeXt backbone trained from scratch is worse.

#### Hyperparameters

Our main model uses 3 refinement blocks, and radius-4 correlations computed at 5 scales. We demonstrate in Table [7](https://arxiv.org/html/2506.07310v2#S4.T7) that the neighboring alternatives are worse. We note that using 6 scales produces a runtime error, as $256\times 256$ input is already $1\times 1$ after 5 scales.

#### Huber vs. L1

CoTracker3 [24] and TAPIR [12] recommend a Huber loss instead of the L1 used in PIPs [19]; in our setup this is not helpful: 54.9 (Huber) vs. 56.8 (L1).

#### Does frame ordering matter?

We train a model with shuffled frames, to disentangle the benefits of joint multi-frame inference from the benefits of temporal continuity: we find that the impact varies from dataset to dataset, but shuffling is worse on average: 56.3 vs. 56.8.

## 5 Conclusion and Limitations

AllTracker is an approach to point tracking that treats the task as multi-frame optical flow.
Our model delivers state-of-the-art performance on point tracking benchmarks, and produces dense output at high resolution, which past point trackers have struggled to do.
AllTracker makes sparse point trackers (of similar speed and accuracy) redundant, but interestingly it does not make optical flow methods redundant: it does not outperform the state-of-the-art optical flow methods on optical flow estimation. The model appears to be underfitting on short-range motion estimation, suggesting that better models might be obtained with greater compute.
A related limitation to address is the temporal window size: with bigger GPUs it may be possible to simply train and test with wider windows [24], and resolve longer occlusions.
A broader area for future work is to add awareness about physical and common-sense constraints on motion, perhaps using 3D [56, 37] or more expressive model designs [18, 60].

#### Acknowledgments

This work was supported by the TRI University 2.0 program, ARL grant W911NF-21-2-0104, and a Vannevar Bush Faculty Fellowship.

## References

- Balasingam et al. [2024]
Arjun Balasingam, Joseph Chandler, Chenning Li, Zhoutong Zhang, and Hari Balakrishnan.
Drivetrack: A benchmark for long-range point tracking in real-world videos.
In *CVPR*, 2024.
- Bian et al. [2024]
Weikang Bian, Zhaoyang Huang, Xiaoyu Shi, Yitong Dong, Yijin Li, and Hongsheng Li.
Context-pips: Persistent independent particles demands context features.
*NeurIPS*, 2024.
- Biggs et al. [2018]
Benjamin Biggs, Thomas Roddick, Andrew Fitzgibbon, and Roberto Cipolla.
Creatures great and SMAL: Recovering the shape and motion of animals from video.
In *ACCV*, 2018.
- Brox and Malik [2010]
Thomas Brox and Jitendra Malik.
Large displacement optical flow: descriptor matching in variational motion estimation.
*TPAMI*, 33(3):500–513, 2010.
- Cho et al. [2024]
Seokju Cho, Jiahui Huang, Seungryong Kim, and Joon-Young Lee.
Flowtrack: Revisiting optical flow for long-range dense tracking.
In *CVPR*, 2024.
- Cho et al. [2025]
Seokju Cho, Jiahui Huang, Jisu Nam, Honggyu An, Seungryong Kim, and Joon-Young Lee.
Local all-pair correspondence for point tracking.
In *ECCV*, 2025.
- Conze et al. [2014]
Pierre-Henri Conze, Philippe Robert, Tomas Crivelli, and Luce Morin.
Dense long-term motion estimation via statistical multi-step flow.
In *VISAPP*, 2014.
- Conze et al. [2016]
Pierre-Henri Conze, Philippe Robert, Tomas Crivelli, and Luce Morin.
Multi-reference combinatorial strategy towards longer long-term dense motion estimation.
*Computer Vision and Image Understanding*, 150:66–80, 2016.
- Crivelli et al. [2012]
Tomas Crivelli, Pierre-Henri Conze, Philippe Robert, and Patrick Pérez.
From optical flow to dense long term correspondences.
In *International Conference on Image Processing*, 2012.
- Darkhalil et al. [2024]
Ahmad Darkhalil, Rhodri Guerrier, Adam W Harley, and Dima Damen.
Egopoints: Advancing point tracking for egocentric videos.
*arXiv:2412.04592*, 2024.
- Doersch et al. [2022]
Carl Doersch, Ankush Gupta, Larisa Markeeva, Adrià Recasens, Lucas Smaira, Yusuf Aytar, João Carreira, Andrew Zisserman, and Yi Yang.
TAP-Vid: A benchmark for tracking any point in a video.
In *NeurIPS Datasets and Benchmarks*, 2022.
- Doersch et al. [2023]
Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman.
TAPIR: Tracking any point with per-frame initialization and temporal refinement.
In *ICCV*, 2023.
- Doersch et al. [2024]
Carl Doersch, Pauline Luc, Yi Yang, Dilara Gokay, Skanda Koppula, Ankush Gupta, Joseph Heyward, Ignacio Rocco, Ross Goroshin, João Carreira, et al.
Bootstap: Bootstrapped training for tracking-any-point.
In *ACCV*, 2024.
- Dosovitskiy et al. [2015]
Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg, Philip Hausser, Caner Hazirbas, Vladimir Golkov, Patrick Van Der Smagt, Daniel Cremers, and Thomas Brox.
Flownet: Learning optical flow with convolutional networks.
In *ICCV*, 2015.
- Geiger et al. [2013]
Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun.
Vision meets robotics: The kitti dataset.
*International Journal of Robotics Research*, 2013.
- Gibson [1950]
James J Gibson.
The perception of the visual world.
1950.
- Greff et al. [2022]
Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, et al.
Kubric: A scalable dataset generator.
In *CVPR*, 2022.
- Harley et al. [2024]
Adam Harley, Yang You, Yang Zheng, Xinglong Sun, Nikhil Raghuraman, Sheldon Liang, Wen-Hsuan Chu, Suya You, Achal Dave, Pavel Tokmakov, et al.
TAG: Tracking at any granularity.
2024.
- Harley et al. [2022]
Adam W Harley, Zhaoyuan Fang, and Katerina Fragkiadaki.
Particle video revisited: Tracking through occlusions using point trajectories.
In *ECCV*, 2022.
- He et al. [2016]
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition.
In *CVPR*, 2016.
- Horn and Schunck [1981]
Berthold KP Horn and Brian G Schunck.
Determining optical flow.
*Artificial intelligence*, 17(1-3):185–203, 1981.
- Ilg et al. [2017]
Eddy Ilg, Nikolaus Mayer, Tonmoy Saikia, Margret Keuper, Alexey Dosovitskiy, and Thomas Brox.
Flownet 2.0: Evolution of optical flow estimation with deep networks.
In *CVPR*, 2017.
- Karaev et al. [2023]
Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht.
Dynamicstereo: Consistent dynamic depth from stereo videos.
*CVPR*, 2023.
- Karaev et al. [2024a]
Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht.
Cotracker3: Simpler and better point tracking by pseudo-labelling real videos.
*arXiv:2410.11831*, 2024a.
- Karaev et al. [2024b]
Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht.
CoTracker: It is better to track together.
In *ECCV*, 2024b.
- Kondermann et al. [2016]
Daniel Kondermann, Rahul Nair, Katrin Honauer, Karsten Krispin, Jonas Andrulis, Alexander Brock, Burkhard Gussefeld, Mohsen Rahimimoghaddam, Sabine Hofmann, Claus Brenner, et al.
The HCI benchmark suite: Stereo and flow ground truth with uncertainties for urban autonomous driving.
In *CVPR Workshops*, 2016.
- Le Moing et al. [2024]
Guillaume Le Moing, Jean Ponce, and Cordelia Schmid.
Dense optical tracking: connecting the dots.
In *CVPR*, 2024.
- Lee et al. [2021]
Alex X Lee, Coline Manon Devin, Yuxiang Zhou, Thomas Lampe, Konstantinos Bousmalis, Jost Tobias Springenberg, Arunkumar Byravan, Abbas Abdolmaleki, Nimrod Gileadi, David Khosid, et al.
Beyond pick-and-place: Tackling robotic stacking of diverse shapes.
In *CoRL*, 2021.
- Li et al. [2024a]
Hongyang Li, Hao Zhang, Shilong Liu, Zhaoyang Zeng, Feng Li, Tianhe Ren, Bohan Li, and Lei Zhang.
TAPTRv2: Attention-based position update improves tracking any point.
In *NeurIPS*, 2024a.
- Li et al. [2024b]
Hongyang Li, Hao Zhang, Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, and Lei Zhang.
TAPTR: Tracking any point with transformers as detection.
In *ECCV*, 2024b.
- Liu et al. [2022]
Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie.
A convnet for the 2020s.
In *CVPR*, 2022.
- Lucas et al. [1981]
Bruce D Lucas, Takeo Kanade, et al.
An iterative image registration technique with an application to stereo vision.
In *IJCAI*, 1981.
- Mathis et al. [2021]
Alexander Mathis, Thomas Biasi, Steffen Schneider, Mert Yuksekgonul, Byron Rogers, Matthias Bethge, and Mackenzie W Mathis.
Pretraining boosts out-of-domain robustness for pose estimation.
In *WACV*, 2021.
- Mayer et al. [2016]
Nikolaus Mayer, Eddy Ilg, Philip Hausser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox.
A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation.
In *CVPR*, 2016.
- Mehl et al. [2023]
Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andrés Bruhn.
Spring: A high-resolution high-detail dataset and benchmark for scene flow, optical flow and stereo.
In *CVPR*, 2023.
- Neoral et al. [2024]
Michal Neoral, Jonáš Šerỳch, and Jiří Matas.
Mft: Long-term tracking of every pixel.
In *WACV*, 2024.
- Ngo et al. [2025]
Tuan Duc Ngo, Peiye Zhuang, Chuang Gan, Evangelos Kalogerakis, Sergey Tulyakov, Hsin-Ying Lee, and Chaoyang Wang.
DELTA: Dense efficient long-range 3d tracking for any video.
In *ICLR*, 2025.
- Niehorster [2021]
Diederick C Niehorster.
Optic flow: a history.
*i-Perception*, 12(6):20416695211055766, 2021.
- Richter et al. [2017]
Stephan R Richter, Zeeshan Hayder, and Vladlen Koltun.
Playing for benchmarks.
In *ICCV*, 2017.
- Sand and Teller [2006]
P. Sand and S. Teller.
Particle video: Long-range motion estimation using point trajectories.
In *CVPR*, 2006.
- Shi et al. [2016]
Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang.
Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network.
In *CVPR*, 2016.
- Sun et al. [2018]
Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz.
PWC-Net: CNNs for optical flow using pyramid, warping, and cost volume.
In *CVPR*, 2018.
- Sun et al. [2021]
Deqing Sun, Daniel Vlasic, Charles Herrmann, Varun Jampani, Michael Krainin, Huiwen Chang, Ramin Zabih, William T Freeman, and Ce Liu.
Autoflow: Learning a better training set for optical flow.
In *CVPR*, 2021.
- Sun et al. [2024]
Xinglong Sun, Adam W Harley, and Leonidas J Guibas.
Refining pre-trained motion models.
In *ICRA*, 2024.
- Sundaram et al. [2010]
Narayanan Sundaram, Thomas Brox, and Kurt Keutzer.
Dense point trajectories by GPU-accelerated large displacement optical flow.
In *ECCV*, 2010.
- Sundararaman et al. [2021]
Ramana Sundararaman, Cedric De Almeida Braga, Eric Marchand, and Julien Pettre.
Tracking pedestrian heads in dense crowd.
In *CVPR*, 2021.
- Teed and Deng [2020]
Zachary Teed and Jia Deng.
RAFT: Recurrent all-pairs field transforms for optical flow.
In *ECCV*, 2020.
- Tolstikhin et al. [2021]
Ilya O. Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, and Alexey Dosovitskiy.
MLP-mixer: An all-mlp architecture for vision.
*arXiv:2105.01601*, 2021.
- Tomasi and Kanade [1991]
Carlo Tomasi and Takeo Kanade.
Detection and tracking of point.
*IJCV*, 9:137–154, 1991.
- Tournadre et al. [2024]
Marc Tournadre, Catherine Soladié, Nicolas Stoiber, and Pierre-Yves Richard.
Dense trajectory fields: Consistent and efficient spatio-temporal pixel tracking.
In *ACCV*, 2024.
- Vaswani et al. [2017]
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.
Attention is all you need.
In *NeurIPS*, 2017.
- Vecerik et al. [2024]
Mel Vecerik, Carl Doersch, Yi Yang, Todor Davchev, Yusuf Aytar, Guangyao Zhou, Raia Hadsell, Lourdes Agapito, and Jon Scholz.
RoboTAP: Tracking arbitrary points for few-shot visual imitation.
In *ICRA*, 2024.
- Wang et al. [2020]
Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer.
TartanAir: A dataset to push the limits of visual slam.
In *IROS*, 2020.
- Wang et al. [2024]
Yihan Wang, Lahav Lipson, and Jia Deng.
Sea-raft: Simple, efficient, accurate raft for optical flow.
In *ECCV*, 2024.
- Wu et al. [2023]
Guangyang Wu, Xiaohong Liu, Kunming Luo, Xi Liu, Qingqing Zheng, Shuaicheng Liu, Xinyang Jiang, Guangtao Zhai, and Wenyi Wang.
Accflow: Backward accumulation for long-range optical flow.
In *ICCV*, 2023.
- Xiao et al. [2024]
Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou.
Spatialtracker: Tracking any 2d pixels in 3d space.
In *CVPR*, 2024.
- Xu et al. [2022]
Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao.
Gmflow: Learning optical flow via global matching.
In *CVPR*, 2022.
- Xu et al. [2017]
Jia Xu, René Ranftl, and Vladlen Koltun.
Accurate optical flow via direct cost volume processing.
In *CVPR*, 2017.
- Zheng et al. [2023]
Yang Zheng, Adam W. Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J. Guibas.
Pointodyssey: A large-scale synthetic dataset for long-term point tracking.
In *ICCV*, 2023.
- Zholus et al. [2025]
Artem Zholus, Carl Doersch, Yi Yang, Skanda Koppula, Viorica Patraucean, Xu Owen He, Ignacio Rocco, Mehdi SM Sajjadi, Sarath Chandar, and Ross Goroshin.
TAPNext: Tracking any point (TAP) as next token prediction.
*arXiv:2504.05579*, 2025.

## Appendix A Additional model details

#### Recurrent module

In the recurrent module, we compress and contextualize the input data in stages [54], following the design ideas of SEA-RAFT [54], as illustrated in Figure [5](https://arxiv.org/html/2506.07310v2#A1.F5).
We use parallel 2-layer CNNs (with $3\times 3$ kernels) on the correlation field and the motion field, then concatenate these feature maps, and merge them with a $1\times 1$ convolution. We then concatenate the visibility map, confidence map, and appearance features, and merge them to $256$ channels with another $1\times 1$ convolution. We then apply three “space-time” blocks, where the spatial part is a 2D ConvNeXt block (with a $7\times 7$ kernel) and the temporal part is a pixel-aligned transformer block (attending the full subsequence span $S$). By pixel-aligned, we mean that attention only happens along the temporal axis (i.e., with cost quadratic in $S=16$), and this is done for every pixel in parallel.
We note that an important convenience of our design is that all of the tensors in this stage are aligned with the “query” frame; therefore pixel-aligned attention is attention between corresponding pixels. After the interleaved spatial and temporal blocks propagate tracking-related information across our window,
we emit a new hidden state, and decode this state into explicit revisions for visibility, confidence, and motion.
Following SEA-RAFT [54], we only use half of the feature channels for our recurrent module’s hidden state, as shown by the “split” step in Figure [5](https://arxiv.org/html/2506.07310v2#A1.F5), which saves memory (reducing the number of output channels from $260$ to $132$) and may also stabilize recurrence.

#### Model layers

The ConvNeXt blocks are standard: layer-scaled kernel-7 grouped convolution $\rightarrow$ layer norm $\rightarrow$ expansion linear layer (factor 4) $\rightarrow$ GELU $\rightarrow$ reduction linear layer (factor 4) $\rightarrow$ residual add $\rightarrow$ linear layer. The pixel-aligned temporal blocks also have a standard form: layer-scaled transformer block with 8 heads and expansion factor 4 $\rightarrow$ residual add $\rightarrow$ linear layer.
To inform the attention layers on frame ordering, we add 1D sinusoidal position embeddings to the “context” features of a subsequence (see Figure 3 from the main paper), broadcasting these embeddings across the spatial axes. Note that our temporal position embedding is with respect to a subsequence; the model unaware of total video length, and we give no information about the anchor frame’s original position in the timeline.

Figure: Figure 5: Detailed view of iterative refinement block. We consolidate data from visibility, confidence, correlation, motion, and appearance features into a single feature map, then interleave convolutional spatial blocks and pixel-aligned temporal blocks, and output revisions to the features, visibility, confidence, and motion. This refinement process is iterated 4 times (with shared weights).
Refer to caption: x5.png

#### AllTracker-Tiny

For AllTracker-Tiny, we use a BasicEncoder backbone with channel dimension 128, making only 2.63M parameters (out of 6.29M total), and leave the rest of the architecture unchanged. We use this model’s featuremap as both the “context” features and the RNN hidden state initialization, rather than splitting a 256-channel featuremap into these two parts.

Figure: Table 8: Comparison against CoTracker3 with different evaluation protocols. We evaluate $\delta_{\text{avg}}$ (higher is better), using an input resolution of $384\times 512$. The benchmarks are BADJA [3], CroHD [46], TAPVid-DAVIS [11], DriveTrack [1], EgoPoints [10], Horse10 [33], TAPVid-Kinetics [11], RGB-Stacking [28], and RoboTAP [52]. “CoTracker3*” indicates values reported in the CoTracker3 paper under a more expensive evaluation protocol (and on fewer datasets). Parameter counts are in millions.

#### Initialization strategy

A close comparison of our model architecture versus SEA-RAFT [54] will reveal that we do not follow SEA-RAFT’s choice to directly regress an “initial” optical flow estimate with a secondary CNN. Our main reason for omitting this is to save memory. We also note that when frame gaps are sufficiently large, it may in fact be impossible to estimate optical flow without temporal context (e.g., out-of-bounds motion of 200 pixels vs. 300 pixels appears identical), and therefore the flow regression from SEA-RAFT would not likely be as effective as simply propagating the estimates from the previous window.

Figure: Figure 6: Visualization of dense correspondence maps produced by all models. On the far left column we show the ground truth trajectories overlaid on the first frame of the input video, with blue-to-green colormap. (Note that a ground truth flow map does not exist in this data.) The flow maps in the other columns show the estimated correspondence field from the first frame of a video to the last frame of the video. Note that RAFT and SEA-RAFT only make use of the first and last frames, while other methods use the intermediate frames as well.
Refer to caption: x6.png

## Appendix B Additional training details

We train with mixed precision in PyTorch (bfloat16).

From point tracking datasets, we use samples which have anywhere upwards of 256 valid annotated tracks (after augmentations), and trim to a maximum of 6144 tracks to keep memory usage predictable. From optical flow datasets we use dense supervision, but note that this data does not include visibility labels.

The major variables in speed and memory consumption are batch size, video length, input resolution, and number of refinement steps. To match inference speed across inputs of different length and keep memory consumption within our budget (8x A100 40G), we use the following settings: on optical flow data (where video length is 2), we use batch size 8, resolution $384\times 768$, and $4$ refinement steps; on videos of length $24$, we use batch size 1, resolution $384\times 512$, and $4$ refinement steps; on videos of length $56$, we use batch size 1, resolution $256\times 384$, and $3$ refinement steps. In the first stage of training (on Kubric alone), we use only videos of length 24 and 56, and split our 8 GPUs by using 4 for the 24-frame videos and 4 for 56-frame videos. In the second stage of training (on the wider mix of data), we split our 8 GPUs by using 1 for optical flow, 3 for videos of length 24, and 4 for videos of length 56. We sync gradients across GPUs after each backward pass.

For our BCE loss, we apply the sigmoid first and then use the direct BCE loss, which we (counter-intuitively) found to be more numerically stable than BCE with logits.

We note that no architecture modifications are required to train jointly for optical flow estimation and point tracking. In optical flow, the temporal attention is a redundant operation, but we do not disable the temporal transformer, as there are still MLP layers within it which participate in the processing.

## Appendix C Additional baseline details

As mentioned in the main paper, the performance of CoTracker-style models depends how the query points are grouped. Intuitively, if multiple queries lie on the same object, they will be tracked more accurately. When these methods are tasked with tracking all of the benchmark queries at once, they tend to exploit a bias in the data and perform better than they would perform on random queries. The authors of these methods suggest a strategy for mitigating this effect, which consists of running each query in a separate pass, while also adding “support” points around the query and a sparse grid covering the image. In our high-resolution multi-benchmark evaluation, these steps would be prohibitively expensive (e.g., weeks). We therefore simply give the models the advantage of the data bias: we give all queries at once, and supplement them with a sparse grid of points around the image. We compare this evaluation protocol to author-reported results in Table [8](https://arxiv.org/html/2506.07310v2#A1.T8). We find that our cheaper protocol over-estimates the accuracy of CoTracker3, but our own model is still more accurate on average.

## Appendix D Additional optical flow results

Figure: Table 9: Optical flow end-point error (“EPE-All”) in the offical SINTEL test benchmark.

Figure: Table 10: Optical flow end-point error in the CVO “Final” (T=7) and “Extended” (T=48) test sets, for visible/occluded pixels.

We ran AllTracker on the official SINTEL test benchmark, yielding the scores shown in Table [9](https://arxiv.org/html/2506.07310v2#A4.T9), with other official scores included for comparison. We note that it is common in optical flow literature to produce a different model for each benchmark (finetuned with a particular data mix and resolution), but we simply use our original checkpoint. AllTracker’s optical flow is not as accurate as SEA-RAFT, but comparable to RAFT or GMFlow.
Qualitatively, AllTracker’s flow maps appear to be coarser than the ones from SEA-RAFT [54], suggesting that the model is underfitting; better models might be obtained with greater compute.

We additionally evaluate in CVO, the multi-frame optical flow dataset used by AccFlow and DOT, showing results in Table [10](https://arxiv.org/html/2506.07310v2#A4.T10). We find that on short sequences DOT (combining CoTracker2 and RAFT) performs best; on long sequences AllTracker performs best. We also find that AllTracker is 3x the speed of DOT.

In sum, these results suggest that AllTracker is not state-of-the-art for optical flow estimation, even though it includes optical flow data in its training. Attaining top performance optical flow and point tracking with a single model remains an open challenge.

## Appendix E Additional ablation details

#### Validation dataset

As mentioned in the main paper, we construct a validation dataset for our ablation studies, using BADJA [3], CroHD [46], TAPVid-Davis [11], DriveTrack [1], Horse10 [33], and RoboTAP [52]. The purpose of these studies is to obtain a quick (but reliable) look at performance, and therefore we do not use these datasets in their entirety, and note we also exclude some of our available datasets. We subsample from these datasets by (1) selecting the first frame with any annotations and tracking only the queries on that frame, (2) trimming all videos to a maximum length of 300 frames. These choices, along with our truncated training regime (training only 100,000 steps and only on Kubric) allows for most ablation experiments to (individually) start and finish within 24 hours.

Figure: Figure 7: Accuracy over inference steps. Accuracy rises quickly then plateaus. In the main evaluation we use 4 iterations.
Refer to caption: x7.png

#### Inference steps

In the main paper we apply the recurrent refinement module 4 times. Figure [7](https://arxiv.org/html/2506.07310v2#A5.F7) shows performance at different iterations, evaluating $\delta_{\textrm{avg}}$ over all datasets and averaging, using an input resolution of $384\times 512$. On the first step the model achieves 62.1 accuracy, which already outperforms most state-of-the-art models. Accuracy rises to its peak at 5 iterations, then begins to drop. In the main paper we report results at 4 iterations, because we find that step 4 and step 5 produce similar accuracy at higher resolutions.

## Appendix F Additional qualitative results

To obtain long-range flow estimates from our point tracker baselines, we query them to track every pixel of the first frame of the video. We perform these queries in “batches” of 10,000, which is the maximum that fits on our GPU. Note that CoTracker3 benefits from processing these jointly, whereas PIPs++ and LocoTrack do not, due to the design of these models.
To obtain long-range flow estimates from the optical flow baselines, we pair the first frame with every other frame, creating $T-1$ frame pairs for flow estimation, where $T$ is the length of the video.

We show additional visualizations of multiple models’ dense outputs in Figure [6](https://arxiv.org/html/2506.07310v2#A1.F6). We notice striking dissimilarity across the outputs of the methods, attesting to the difficulty of the task, and to the usefulness of visualizing point tracks as flow maps. We find that when the foreground displacements are large, the flow models often “give up” on the dynamic foreground and produce a motion field that only describes the background. We also find that PIPs++ and LocoTrack often struggle with spatial smoothness, while the flow models do not. CoTracker3 occasionally fails on smoothness too (see row 3 with the car), but less so. Our model appears to produce results that are smooth and accurate, which matches intuitions for a model that blends the 2D processing of flow models with the temporal coherence of point trackers.