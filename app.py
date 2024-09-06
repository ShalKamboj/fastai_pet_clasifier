import gradio as gr
from fastai.vision.all import *
#import skimage
from PIL import Image

learn = load_learner('export.pkl')

labels = learn.dls.vocab

def predict(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Pet Breed Classifier"
description = "A pet breed classifier trained on the Oxford Pets dataset with fastai. Created as a demo for Gradio and HuggingFace Spaces."
article="<p style='text-align: center'><a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial' target='_blank'>Blog post</a></p>"
examples = ['siamese.jpg', 'snowflake.jpg', 'cleo.jpg', 'cleo2.jpg', 'cleo3.jpg']

gr.Interface(
    fn=predict,
    inputs=gr.components.Image(
        type="numpy",
        height=512,
        width=512,
        label="Upload Image or take a photo",),
    outputs=gr.components.Label(num_top_classes=3),
    title=title,
    description=description,
    article=article,
    examples=examples).queue().launch()
