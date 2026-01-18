# HeartMuLa_ComfyUI
ComfyUI Custom Node for HeartMuLa AI Music Generation and Transcript Text

**HeartMuLa** official GITHUB
https://github.com/HeartMuLa/heartlib


------------------------------------------------------------

# Installation

------------------------------------------------------------

**Step 1**

Go to ComfyUI\custom_nodes
Command prompt:

git clone https://github.com/benjiyaya/HeartMuLa_ComfyUI

**Step 2**

cd /HeartMuLa_ComfyUI

**Step 3**

pip install -r requirements.txt

------------------------------------------------------------

# For File structure

------------------------------------------------------------

<img width="1179" height="345" alt="image" src="https://github.com/user-attachments/assets/5087e10e-9815-48ff-bbb4-3a21dc1e54d1" />


------------------------------------------------------------

# Download model files

------------------------------------------------------------
Go to ComfyUI/models 

Use HuggingFace Cli donwload model weigths.

type :

hf download --local-dir './HeartMuLa' 'HeartMuLa/HeartMuLaGen'

hf download --local-dir './HeartMuLa/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-oss-3B'

hf download --local-dir './HeartMuLa/HeartCodec-oss' 'HeartMuLa/HeartCodec-oss'

hf download --local-dir './HeartMuLa/HeartTranscriptor-oss' 'HeartMuLa/HeartTranscriptor-oss' 


------------------------------------------------------------

# For Model File structure

------------------------------------------------------------


<img width="1391" height="320" alt="image" src="https://github.com/user-attachments/assets/3b48ff70-2a4f-4f8d-aed2-d0fbc76bb31f" />



------------------------------------------------------------


Model Sources
------------------------------------------------------------

Github Repo: https://github.com/HeartMuLa/heartlib

Paper: https://arxiv.org/abs/2601.10547

Demo: https://heartmula.github.io/

HeartMuLa-oss-3B: https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B

HeartCodec-oss: https://huggingface.co/HeartMuLa/HeartCodec-oss

HeartTranscriptor-oss: https://huggingface.co/HeartMuLa/HeartTranscriptor-oss


Credits
------------------------------------------------------------
HeartMuLa: https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B


