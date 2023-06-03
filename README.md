# ImageCaptioningWebapp

This is a webapp created using Python and Flask to caption images uploaded by user. We use 6 models to generate captions (1 per model). The models are pretrained and available on huggingface hub. The model names are available in variable "model_name" in app.py file. The execution and result obtaining can take some time after the image is uploaded as the models are large in size. I have tried my best to make the code execute quickly.

The working of the webapp can be viewed in this video: https://drive.google.com/file/d/1gDlggTQi24Ki6Cu4ngxQ4tljR_Q1T2Wn/view?usp=sharing

Steps to run the app:
<ol>
  <li>
    Install the following libraries: pytorch, transformers, flask, Pillow using the following command:
    <code>pip install transformers Pillow torch torchvision torchaudio Flask</code>
  </li>
  <li>
    In order to quicken the execution, run the <code>DownloadModels.py</code> file to download the models, weights and other requirements so that they can be used directly while the webapp is running.
  </li>
  <li>
    Go to the project directory containing <code>app.py</code> file using <code>cd</code> or <code>ls</code> command.
  </li>
  <li>
    Run the command <code>flask run</code>
  </li>
</ol>
