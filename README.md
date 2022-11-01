instrument_pose_estimation
=============


Models
-------------
 - "Articulated multi-instrument 2-D pose estimation using fully convolutional networks" 구현 [논문 링크](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6051486/)
 
Dataset
-------------
- Endovis dataset [link](https://www.notion.so/Task-Endovis-dataset-a7156a98b4bb4a67b377a3038a242051)
 
Post Processing
-------------
 - Multi instrument parsing (bottom up method)
 1. Joint, Joint pair 를 찾는 알고리즘
 2. 찾은 Joint를 parsing 하는 부분

Training, Testing, Inference
-------------
training
<pre><code>python train.py configs/train.py</code></pre> 

testing
<pre><code>python test.py configs/test.py</code></pre> 

inference
<pre><code>python inference.py configs/inference.py</code></pre> 
