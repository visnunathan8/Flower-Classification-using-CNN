
<h1> FlowerSpeciesClassification </h1>

• Accurate identification and classification of flower species is a crucial task for the understanding and conservation of various plant species. <br>
• However, the lack of available information about the different flower species poses a significant challenge to achieving this goal. <br>
• This report proposes a Flower Identification System focused on identifying the correct class/specie of a given flower. <br>
• The project utilizes three different datasets containing a varied number of flower species, that were used to train three different Convolutional Neural • Network (CNN) architectures - MobileNetV2, ResNet18, and VGG16. <br>
• The training process was done for nine instances trained from scratch, three instances for transfer learning and multiple instances for hyperparameter tuning. <br>
• In this report, the outcomes of the  scratch and transfer learning training, hyperparameter tuning, Grad-CAM and TSNE visualization are discussed.<br>
• The analysis helps determine which model-dataset combination provides the maximum accuracy along with optimal hyperparameters for training the models.<br> 
• This study aims to contribute towards enhancing the classification and identification of flower species, which will be beneficial for the conservation and protection of various plant species.<br>

<h2> Project aim </h2> • Develop a flower classification system using deep learning techniques to aid botanists, agriculturists, and horticulturists in identifying different species of flowers.<br>
<h2> Methodology </h2> • Building deep-learning-based classification models using a combination of state-of-the-art convolutional neural network (CNN) architectures. <br>
 <h2> Challenges </h2>
    • Differences in image qualities within the dataset. <br>
    • Data quality issues, where some images contained text or the flower was not the most prominent part of the image. <br>
<h2> Goals </h2>
    • Explore and provide a detailed analysis of how different CNN architectures and combinations fare against the chosen datasets of flowers. <br>
    • Compare all eleven models that were trained. <br>
    • Provide a detailed performance analysis. <br>

<h2> Requirements to run the code (libraries, etc) </h2>
 
 ```sh
  pip install -r requirements.txt
  ```
 
<h3> Instructions to run the code : </h3>
• Jupyter Notebook or any compatible software to run .ipynb files. <br>
• Access to the dataset, which is assumed to be stored in Google Drive. <br>
• Access to the pre-trained models, which are stored inside PreTrainedModels. <br>
• The required libraries and modules should be installed in the environment, including but not limited to PyTorch, scikit-learn, tqdm, and matplotlib. <br>

<h2>Training and Validating the Models</h2>

The models are trained in the folders as in Dataset-1, Dataset-2, Dataset-3.


<h2>To run the pre-trained model on the provided sample test dataset </h2> 
• Access the "TestingModel" folder where the pre-trained sample testing model is located. <br>
• Retrieve the corresponding pre-trained model weights from the below given drive link. <br>
• Obtain the sample test dataset from the below given drive link. (https://drive.google.com/drive/folders/1BrCI3fdoxvH840Ii5AD914SgKfVjv8Ci?usp=share_link) <br>


<h2> Data Preprocessing </h2>

<h2>Links to the Dataset</h2>
• Flowers Dataset 1, URL: https://www.kaggle.com/datasets/nadyana/flowers <br>
• Flowers Dataset 2, URL: https://www.kaggle.com/datasets/utkarshsaxenadn/flower-classification-5-classes-roselilyetc <br>
• Flowers Dataset 3, URL: https://www.kaggle.com/datasets/l3llff/flowers <br>

<br>
<img width="700" alt="Screenshot 2023-04-12 at 12 37 25 AM" src="https://user-images.githubusercontent.com/30067377/231351224-b85335c2-027d-47ba-aeb3-a5b92d1a6bd3.png">

![std2](https://user-images.githubusercontent.com/30067377/231942056-2a3c504c-c9ba-4346-8294-b73b69d5e27b.png)

<h3>Dataset-1 </h3> 
  • It has 7 evenly balanced classes with 1600 images per class (11200 total images). However, we pruned the number of classes to 5 making the total images to 8000(to maintain a diverse number of classes per dataset). 
  
<img width="905" alt="Screenshot 2023-04-12 at 2 13 04 AM" src="https://user-images.githubusercontent.com/30067377/231367104-14256f00-be8a-4684-968c-7f70ed518342.png">

<h3>Dataset-2 </h3> 
  • It has 10 evenly distributed classes with 1500 images per class (15000 total images).
  
<img width="905" alt="Screenshot 2023-04-12 at 2 13 42 AM" src="https://user-images.githubusercontent.com/30067377/231367205-7fe78ffb-0c40-4ef8-bf3c-dc59e668a8a0.png">


<h3>Dataset-3 </h3> 
  • It has 16 classes unevenly distributed with a total of 15,740 images. There are 980 images on average per class in Dataset-3 where the number of images per class fell in the range of 737-1054. 
  
  <img width="905" alt="Screenshot 2023-04-12 at 2 14 05 AM" src="https://user-images.githubusercontent.com/30067377/231367296-2d956e6a-c479-4881-892d-5ed5bd2d4642.png">

<h3> Problematic Images in Dataset 2 </h3>

<img width="905" alt="Screenshot 2023-04-12 at 2 15 01 AM" src="https://user-images.githubusercontent.com/30067377/231367467-287da706-15e2-43bf-ad24-d5da7eebac71.png">

<br> <br>


<h3> The imported libraries and modules include : </h3>

* [![Python][Python.js]][Python-url]
* [![torch][torch.js]][torch-url]
* [![torch.nn, torch.nn.functional][torch.nn.js]][torch.nn-url]
* [![torch.utils.data][torch.utils.data.js]][torch.utils.data-url]
* [![torchvision.datasets, torchvision.transforms, torchvision.models][torchvision.js]][torchvision-url]
* [![sklearn][sklearn.js]][sklearn-url]
* [![tqdm][tqdm.js]][tqdm-url]
* [![torchsummary][torchsummary.js]][torchsummary-url]
* [![pandas][pandas.js]][pandas-url]
* [![numpy][numpy.js]][numpy-url]
* [![matplotlib][matplotlib.js]][matplotlib-url]
* [![omnixai][omnixai.js]][omnixai-url]
* [![optuna][optuna.js]][optuna-url]

[Python.js]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[torch.js]: https://img.shields.io/badge/torch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[torch-url]: https://pytorch.org/
[torch.nn.js]: https://img.shields.io/badge/torch.nn%2C%20torch.nn.functional-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[torch.nn-url]: https://pytorch.org/docs/stable/nn.html
[torch.utils.data.js]: https://img.shields.io/badge/torch.utils.data-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[torch.utils.data-url]: https://pytorch.org/docs/stable/data.html
[torchvision.js]: https://img.shields.io/badge/torchvision.datasets%2C%20torchvision.transforms%2C%20torchvision.models-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[torchvision-url]: https://pytorch.org/vision/
[sklearn.js]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[sklearn-url]: https://scikit-learn.org/stable/
[tqdm.js]: https://img.shields.io/badge/tqdm-4B8BBE?style=for-the-badge&logo=python&logoColor=white
[tqdm-url]: https://tqdm.github.io/
[torchsummary.js]: https://img.shields.io/badge/torchsummary-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[torchsummary-url]: https://github.com/sksq96/pytorch-summary
[pandas.js]: https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[pandas-url]: https://pandas.pydata.org/
[numpy.js]: https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white
[numpy-url]: https://numpy.org/
[matplotlib.js]: https://img.shields.io/badge/matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white
[matplotlib-url]: https://matplotlib.org/
[omnixai.js]: https://img.shields.io/badge/omnixai-1C263F?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgo
[omnixai-url]: https://github.com/salesforce/OmniXAI
[optuna.js]: https://optuna.org/assets/img/optuna-logo@2x.png
[optuna-url]: https://optuna.org/


• There are also installations of the optuna and omnixai libraries using the !pip install command. <br>

<h2> Collaborators to our Project </h2>
• GitHub ID: mahdihosseini, email tied to GitHub: (mahdi.hosseini@mail.utoronto.ca). <br>
• GitHub ID: ahmedalagha1418, email tied to GitHub: (ahmedn.alagha@hotmail.com). <br>
• GitHub ID: visnunathan8, email tied to GitHub: (rocketvisnu@gmail.com). <br>
• GitHub ID: ShrawanSai, email tied to GitHub: (msaishrawan@gmail.com).<br>
• GitHub ID: Sharanyu, email tied to GitHub: (sharanyu@hotmail.com).<br>
• GitHub ID: kin-kins, email tied to GitHub: (K.ashu403@gmail.com).
<br><br>

[![AhmedAlagha1418][contributors-shield1]][contributors-url1]
[![mahdihosseini][contributors-shield2]][contributors-url2]


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield1]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-shield2]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge


[contributors-url1]:https://github.com/AhmedAlagha1418
[contributors-url2]: https://github.com/mahdihosseini



