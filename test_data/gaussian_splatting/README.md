---
license: apache-2.0
language:
- en
pretty_name: FiftyOne_Gaussian_Splatting
---

## **Gaussian Splats Dataset**  
**3D Gaussian Splatting for Real-Time Radiance Field Rendering**  

**Dataset Author**: Paula Ramos  
**Created Using**: [3D Gaussian Splatting Paper](https://arxiv.org/abs/2308.04079)  
**Code Repository**: [GitHub - graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) 


## **Description**

This dataset consists of Gaussian Splats representations of different real-world scenes, created using the official 3D Gaussian Splatting method. Each scene folder contains:

    A reference image representing the scene.
    A PLY file stored in a point_cloud_folder, containing the Gaussian Splats reconstruction.


### **Overview**  
This dataset consists of **Gaussian Splats representations** of different real-world scenes, created using the official **3D Gaussian Splatting method**. Each scene folder contains:  
- A **reference image** representing the scene.  
- Two **PLY files** stored in a `point_cloud_folder`, containing the **Gaussian Splats reconstructions at iterations 7000 and 30000**.  

The dataset is structured as follows:
```plaintext
FO_dataset/
│── drjohnson/          # Scene Folder
│   ├── reference_image.png
│   ├── point_cloud_folder/
│       ├── reconstruction_7000.ply
│       └── reconstruction_30000.ply
│── playroom/
│   ├── reference_image.png
│   ├── point_cloud_folder/
│       ├── reconstruction_7000.ply
│       └── reconstruction_30000.ply
│── train/
│   ├── reference_image.png
│   ├── point_cloud_folder/
│       ├── reconstruction_7000.ply
│       └── reconstruction_30000.ply
│── truck/
│   ├── reference_image.png
│   ├── point_cloud_folder/
│       ├── reconstruction_7000.ply
│       └── reconstruction_30000.ply
```

---

## **How to Use the Dataset**  
### **1. Install the Required FiftyOne Plugin**  
To visualize all `.ply` files using FiftyOne, download the **Gaussian Splats plugin**:
```bash
!fiftyone plugins download https://github.com/danielgural/ksplats_panel
```

### **2. Load & Visualize the Dataset with FiftyOne**  
Use the following Python script to **load and explore the dataset** in FiftyOne:

```python
import fiftyone as fo
from fiftyone.utils.splats import SplatFile

# Create a FiftyOne dataset
dataset = fo.Dataset(name="splat-test", overwrite=True)

# Add samples (update paths as needed)
sample1 = fo.Sample(filepath="FO_dataset/drjohnson/reference_image.png")
sample1["splat"] = SplatFile(filepath="FO_dataset/drjohnson/point_cloud_folder/reconstruction_30000.ply")

sample2 = fo.Sample(filepath="FO_dataset/drjohnson/reference_image.png")
sample2["splat"] = SplatFile(filepath="FO_dataset/drjohnson/point_cloud_folder/reconstruction_7000.ply")

sample3 = fo.Sample(filepath="FO_dataset/playroom/reference_image.png")
sample3["splat"] = SplatFile(filepath="FO_dataset/playroom/point_cloud_folder/reconstruction_7000.ply")

sample4 = fo.Sample(filepath="FO_dataset/playroom/reference_image.png")
sample4["splat"] = SplatFile(filepath="FO_dataset/playroom/point_cloud_folder/reconstruction_30000.ply")

sample5 = fo.Sample(filepath="FO_dataset/train/reference_image.png")
sample5["splat"] = SplatFile(filepath="FO_dataset/train/point_cloud_folder/reconstruction_7000.ply")

sample6 = fo.Sample(filepath="FO_dataset/train/reference_image.png")
sample6["splat"] = SplatFile(filepath="FO_dataset/train/point_cloud_folder/reconstruction_30000.ply")

sample7 = fo.Sample(filepath="FO_dataset/truck/reference_image.png")
sample7["splat"] = SplatFile(filepath="FO_dataset/truck/point_cloud_folder/reconstruction_7000.ply")

sample8 = fo.Sample(filepath="FO_dataset/truck/reference_image.png")
sample8["splat"] = SplatFile(filepath="FO_dataset/truck/point_cloud_folder/reconstruction_30000.ply")

# Add samples to the dataset
dataset.add_sample(sample1)
dataset.add_sample(sample2)
dataset.add_sample(sample3)
dataset.add_sample(sample4)
dataset.add_sample(sample5)
dataset.add_sample(sample6)
dataset.add_sample(sample7)
dataset.add_sample(sample8)

# Launch FiftyOne App
session = fo.launch_app(dataset, auto=False, port=5152)
```
---
## **Visualization Results**
Below are sample screenshots showcasing the **3D Gaussian Splats reconstructions**:
![Image](https://github.com/user-attachments/assets/7b829255-b61b-4db8-b21c-dcc73796100a)

### **Drjohnson Scene**
![Image](https://github.com/user-attachments/assets/b580e602-9619-4f59-bc6d-95aa6cdeecd7)

### **Playroom Scene**
![Image](https://github.com/user-attachments/assets/989e3e2a-5be5-4ba0-bea6-2db3d2c16fad)
https://github.com/user-attachments/assets/1c3d3b6b-2b7b-4e93-8f5c-76a184f51260

### **Train Scene**
![Image](https://github.com/user-attachments/assets/a50f013d-e263-4e45-82c9-40f1f46ed24f)
https://github.com/user-attachments/assets/78ca63f5-1df9-4970-a50c-bfab0ee3615f

### **Truck Scene**
![Image](https://github.com/user-attachments/assets/89094b67-85da-4b16-814b-0dc65270ff5a)


---

## **Research & Applications**  
This dataset is useful for a variety of **3D vision and AI applications**, including:  
- **NeRF & Gaussian Splatting Benchmarking**  
- **3D Scene Understanding & Reconstruction**  
- **Multi-Modal AI (Images + 3D Point Clouds)**  
- **Real-Time 3D Rendering Research**  

---

## **Citation**  
If you use this dataset, please cite the original **3D Gaussian Splatting** paper:
```bibtex
@article{kerbl2023gsplatting,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimkühler, Thomas and Drettakis, George},
  journal={arXiv preprint arXiv:2308.04079},
  year={2023}
}
```


And also the link of this dataset in hugging Face: https://huggingface.co/datasets/pjramg/gaussian_splatting/
---



