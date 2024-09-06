from fastai.vision.all import *

path = untar_data(URLs.PETS)

additional_images = Path('images')
ims = get_image_files(additional_images)
for i, img in enumerate(ims):
    category = img.parent.name
    try:
        im = PILImage.create(img)
    except:
        print(f"Failed to open {img}")
        continue

    shutil.copy(src=img, dst=path/'images'/f"{category}_{i}.jpg")

failed = verify_images(path/'images')
print(f"Failed images: {failed}")
failed.map(Path.unlink)

dls = ImageDataLoaders.from_name_re(
    path, 
    get_image_files(path/'images'), 
    pat='(.+)_\d+.jpg', 
    item_tfms=RandomResizedCrop(460, min_scale=0.5),
    # item_tfms=Resize(460), 
    batch_tfms=aug_transforms(size=224, min_scale=0.75))

learn = vision_learner(
    dls, 
    models.resnet50, 
    metrics=accuracy)

print(learn.dls.vocab)

learn.fine_tune(1)
learn.path = Path('.')
learn.export()

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

interp.plot_top_losses(50, nrows=10)