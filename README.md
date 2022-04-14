This is the code for a CVPR 2019 paper, called 
[GoodNews Everyone! Context driven entity aware captioning for news images](https://arxiv.org/abs/1904.01475). Enjoy!

Model preview:

![GoodNews Model!](https://github.com/furkanbiten/GoodNews/blob/master/model.jpg)

Huge Thanks goes to [New York Times API](https://developer.nytimes.com/indexV2.html) for providing such a service for FREE!

Another Thanks to [@ruotianluo](https://github.com/ruotianluo) for providing the captioning code.

Dependencies/Requirements:
```text
pytorch==1.0.0
spacy==2.0.11
h5py==2.7.0
bs4==4.5.3
joblib==0.12.2
nltk==3.2.3
tqdm==4.19.5
urllib2==2.7
goose==1.0.25
urlparse
unidecode
```

# Introduction  
We took the first steps to move the captioning systems to interpretation (see the paper for more detail). 
To this end, we have used [New York Times API](https://developer.nytimes.com/indexV2.html) 
to retrieve the articles, images and captions. 

The structure of this repo is as follows:
1. Getting the data 
2. Cleaning and formating the data
3. How to train models

# Get the data
You have 3 options to get the data. 

## Images only
If you want to download the images only and directly start working on the same dataset as ours, 
then download the cleaned version of the dataset without images: 
[article+caption.json](https://cvcuab-my.sharepoint.com/:u:/g/personal/abiten_cvc_uab_cat/ERuh81o1cnJJoKne8hOe3MYBcFlTglof3vxCcfbyXa7z-w?e=jtQU0Z) 
and put it to data/ folder and 
download the [img_urls.json](https://cvcuab-my.sharepoint.com/:u:/g/personal/abiten_cvc_uab_cat/Ef0W_O-HU59Pn22b9Bni5oABLDtkBUFFMXiN5cl2vnQxFg?e=75fpWW)
and put it in the `get_data/get_images_only/` folder.

Then run  
```bash
python get_images.py --num_thread 16
```
Then, you will get the images. After that move to ``Clean and Format Data`` section.

PS: I have recieved numerous emails regarding some of the images not present/broken in the img_urls.json. Which is why I decided to put the images on the drive to download in the name of open science.
[Download all images](https://cvcuab-my.sharepoint.com/:u:/g/personal/abiten_cvc_uab_cat/EQpCxX8PMr1NmAfMwgn0YzcBc_LLSWCJ8fTvqy-Agjz_Kw?e=pmIxfz)

## Images + articles
If you would like the get the raw version of the article and captions to do your own cleaning and processing, 
no worries! First download the [article_urls](https://cvcuab-my.sharepoint.com/:u:/g/personal/abiten_cvc_uab_cat/EU2g3r144q1LkPLgf3TFFKYB6PznVxZFFOn51kM1ARrMMg?e=8xfQQ3) 
and go to folder ``get_data/with_article_urls/`` and run 
```bash
python get_data_with_urls.py --num_thread 16
python combine_dataset.py 
```
This will get you the raw version of the caption, articles and also the images. 
After that move to ``Clean and Format Data`` section.

## I want more!
As you know, New York Times is huge. Their articles starts from 1881 (It is crazy!) until well today. 
So in case you want to get ALL the data or expand the data to more years, then first step is go to 
[New York Times API](https://developer.nytimes.com/indexV2.html) and get an API key. All you have to do is just sign up for the API key.

Once you have the key go to folder ``get_data/with_api/`` and run
```bash
python retrieve_all_urls.py --api-key XXXX --start_year XXX --end_year XXX 
```
This is for getting the article urls and then saving in the format of ``month-year``. 
Once you have the all urls from the API, then you run 
```bash
python get_data_api.py
python combine_dataset.py
```
```get_data_api.py``` retrieves the articles, captions and images. 
``combine_dataset.py`` combines yearly data into one file after removing data points 
if they have corrupt image, empty articles or empty captions. After that move to ``Clean and Format Data`` section.

## Small Note
I also provide the links to images and their data splits (train, val, test). 
Even though I always use random seed to decide the split, just in case 
If the GODS meddles with the random seed, here is the link to a json where you can find each image and its split: 
[img_splits.json](https://cvcuab-my.sharepoint.com/:u:/g/personal/abiten_cvc_uab_cat/EUn6ay_SiMZNpByb5JRnuJEBoxBO8QaXurpS8liR_1Orew?e=6f0axr)

# Clean and Format the Data
Now that we have the data, it is time to clean, preprocess and format the data. 

## Preprocess
When you reach this part, you must have ``captioning_dataset.json `` in your ``data/`` folder. 
You can also download [captioning_dataset.json](https://cvcuab-my.sharepoint.com/:u:/g/personal/abiten_cvc_uab_cat/ERuh81o1cnJJoKne8hOe3MYBcFlTglof3vxCcfbyXa7z-w?e=LhsTl6).
You can also download [news_dataset.json](https://cvcuab-my.sharepoint.com/:u:/g/personal/abiten_cvc_uab_cat/ESCMiY_9KrFNkibasc-g6xABTwm8RWMNYIAj0BIN4Oya-w?e=kFqxq9g).
### Captions
This part is for cleaning the captions (tokenizing, removing non-ascii characters, etc.),
 splitting train, val, and test and creating anonymize captions. 

In other words, we change the caption "Alber Einstein taught in Princeton in 1926" to "PERSON_ taught in ORGANIZATION_ in DATE_."
Move to ```preprocess/``` folder and run
```bash
python clean_captions.py
```
### Resize Images
To resize the images to ``256x256``:
```bash
python resize.py --root XXXX --img_size 256
```
### Articles
Get the article format that is needed for the encoding methods by running: ``create_article_set.py``
```bash
python create_article_set.py
```

## Format

Now to create H5 file for captions, images and articles, 
just need to go to ``scripts/`` folder and run **in order**
```bash
python prepro_labels.py --max_length 31 --word_count_threshold 4
python prepro_images.py
```

We proposed 3 different article encoding method. You can download each of encoded article methods, 
[articles_full_avg_](https://cvcuab-my.sharepoint.com/:u:/g/personal/abiten_cvc_uab_cat/ERvJZ-9tWN5MvDZnvmAYbw8B0OeteXqSuIfCwr3ZjeGtUQ?e=pT8J4y),
[articles_full_wavg](https://cvcuab-my.sharepoint.com/:u:/g/personal/abiten_cvc_uab_cat/EXe2OI3HCLVJrasj51R84BEBRDPFtmhNcbz8kZ0kIDcFNQ?e=knLeeT),
[articles_full_TBB](https://cvcuab-my.sharepoint.com/:u:/g/personal/abiten_cvc_uab_cat/EY2NsX7aTpNOgGloYxFihtcB9X0NlSfuy97Dc2_766X3Ag?e=MRXcHs).

Or you can use the code to obtain them:
````bash
python prepro_articles_avg.py
python prepro_articles_wavg.py
python prepro_articles_tbb.py
````

# Train 

Finally we are ready to train. Magical words are:
````bash
python train.py --cnn_weight [YOUR HOME DIRECTORY]/.torch/resnet152-b121ed2d.pth 
````
You can check the ``opt.py`` for changing a lot of the options such dimension size, different models, 
hyperparameters, etc.

# Evaluate
After you train your models, you can get the score according commonly used metrics: Bleu, Cider, Spice, Rouge, Meteor.
Be sure to specify model_path, cnn_model_path, infos_path and sen_embed_path when runing ``eval.py``.
``eval.py`` is usually used in training but it is necessary to run it to get the insertion.
# Insertion
Last but not least ``insert.py``. After you run ``eval.py``, it will produce you a json file with the ids
and their template captions. To fill the correct named entity, you have to run ``insert.py``:

````bash
python insert.py --output [XXX] --dump [True/False] --insertion_method ['ctx', 'att', 'rand']
````
PS: I have been requested to provide model's output, so I thought it would be best to share it with everyone.
[Model Output](https://cvcuab-my.sharepoint.com/:f:/g/personal/abiten_cvc_uab_cat/Eu637xtIZN9NltruagxqDLcBWs-wXCM_kMDac82x0QNBxg?e=2WpmJL)
In this folder, you have:

test.json: Test set with raw and template version of the caption.

article.json: Article sentences which is needed in the ``insert.py``.

w/o article folder: All the models output on template captions, without articles.

with article folder: Our models output in the paper with sentence attention(sen_att) and image attention(vis_att), provided in the json. Hope this is helpful to more of you.



# Conclusion
Thank you and sorry for the bugs!
