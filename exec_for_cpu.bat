powershell -Command "(new-object System.Net.WebClient).DownloadFile('https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5','./mask_rcnn_coco.h5')"&call activate.bat
call pip install -r requirements.txt

cd code
python train.py train --dataset=../dataset/ --weights=coco
