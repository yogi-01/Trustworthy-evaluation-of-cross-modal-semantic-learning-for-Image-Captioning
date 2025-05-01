
Trustworthy Evaluation for Cross Modal Semantic Learning for Image Captioning

Code Link: https://colab.research.google.com/drive/1ciES7duDaxb39obbQwuJsSfVH7MODmPa?usp=sharing

The model of the project is a python based application that generates captions for images. It bridges the connection between Computer Vision and Natural Langauge Processing. The main part of the project is to evaluate trustworthy principles mainly robustness and privacy.

Overview
This project extends a deep learning-based image captioning system into the domain of Trustworthy AI by evaluating the model’s behavior under conditions that test its robustness and privacy.

While the objective of the model was to build a caption generation model using a VGG16 encoder and LSTM decoder, the extension of it centers on the trustworthy evaluation of this model. The key aim is to assess how reliable, stable, and safe the model's outputs are when subjected to various input perturbations and repeated use cases.

The trustworthy evaluation focuses on two key dimensions:
1. Robustness: Tests the model’s sensitivity to image distortions like Gaussian noise, blur, and object occlusion. Captions should remain semantically consistent despite these changes if the model is robust.
2. Privacy: Measures output variance across multiple trials with the same input image. A low variance in output probabilities indicates that the model is not leaking sensitive internal states and is consistent in its predictions.


Features
Targeted Perturbation Tests:
  - Evaluates model stability against Gaussian noise, motion blur, and partial occlusion.
  - Identifies whether visual distortions lead to misleading or incorrect captions.

Caption Degradation Monitoring:
  - Compares original captions with those generated from perturbed images.
  - Detects semantic drift (e.g., caption changing from “a dog playing” to “a boy running”).

Multi-Trial Caption Consistency:
  - Runs the captioning model **multiple times** on the same image to assess variance in output.
  - Captions and output probabilities are logged and analyzed to detect internal randomness or inconsistencies.

Variance-Based Privacy Metric:
  - Captures the variance of model confidence scores across runs to indicate privacy/stability.
  - High variance suggests model unpredictability and potential instability in production. 

Modular Evaluation Code:
  - Evaluation logic is separated from training logic for reuse and easy updates.
  - New distortion types or evaluation metrics can be plugged in easily.


Project Structure
├── .ipynb_checkpoints #will be created when file is run in Jupyter 
├── Input  
     ├── Images                   
     ├── Captions                                 
├── Features.pkl  #will be created after extraction of features           
├── Project_code            
├── requirements.txt    
├──readme.txt


Requirements
- Python
- Google Colab or Jupyter Notebook
- modules
 - tensorflow
 - numpy
 - matplotlib
 - pillow
 - tqdm
 - nltk
 - pickle  

Dataset
Flickr8k dataset 
dataset kaggle link: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

Dataset Setup
1. Download the dataset manually from the above link or use the Kaggle API to download it directly.
2. Once downloaded, extract the dataset so that the Flickr_8k_Dataset folder (containing images) and Flickr8k.token.txt (captions file) are in the root directory of the    project.
3. The folder structure should look like this:
   project_root
      ├──Dataset
	├── Flickr_8k_Dataset/  # Folder containing images
	├── Flickr8k.token.txt  # Captions file
      ├── your_script.py

Setup
1. Clone or download the project repository.

2. Install the required Python packages:
   Use the following command in your terminal or Colab notebook: pip install -r requirements.txt

3. Download the Flickr8k dataset:
- Link: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
- Extract the contents so that the images and captions file are accessible under:
  Dataset/
  ├── Flickr_8k_Dataset/
  └── Flickr8k.token.txt

4. (Optional for Google Colab)
- Mount Google Drive if using Colab:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

5. Open the notebook `cleaned_notebook.ipynb` in Jupyter Notebook or Google Colab.

Usage
1. Verify Dataset and Dependencies
 - Ensure that the Flickr 8k Dataset is correctly placed in the root directory as mentioned in the dataset setup.
 - Install the necessary dependencies using:
	pip install -r requirements.txt

2. Run the Notebook
  -Open the Notebook in the google colab or Jupyter Notebook and run the notebook cell by cell

3. Train the Captioning Model
   - Run all the cells in `cleaned_notebook.ipynb`.
   - This includes loading the dataset, preprocessing, extracting image features with VGG16, tokenizing text, and training the encoder-decoder model.

4. Generate Captions for New Images
   - After training, use the `generate_caption("image_path")` function to produce captions:
     ```python
     caption = generate_caption("Flickr_8k_Dataset/example.jpg")
     print(caption)
     ```

5. Run Trustworthy Evaluation
   - Use the final section of the notebook to evaluate the model’s robustness and privacy.
   - Apply distortions (e.g., blur, noise, occlusion) to test how the model’s caption changes.
   - Run multiple trials on the same image and calculate output variance to assess caption stability.

6. Interpret Results
   - Examine visualizations, BLEU score outputs, and caption consistency.
   - Review whether captions remain semantically stable under perturbations and over repeated runs.


References
 -https://www.youtube.com/watch?v=fUSTbGrL1tc&pp=ygUmaW1hZ2UgY2FwdGlvbiBnZW5lcmF0b3IgcHl0aG9uIHByb2plY3Q%3D
 -https://www.geeksforgeeks.org/vgg-16-cnn-model
 -https://www.sciencedirect.com/science/article/abs/pii/S0167278919305974
 -https://www.sciencedirect.com/science/article/pii/S0925231223004101
 -https://www.cs.cornell.edu/~shmat/shmat_oak17.pdf
 -https://www.sciencedirect.com/science/article/abs/pii/S0045790624005706
 -https://pyimagesearch.com/2021/03/01/adversarial-attacks-with-fgsm-fast-gradient-sign-method/











             


