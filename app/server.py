"""
Server
Code adapted from: https://github.com/render-examples/fastai-v3
License: Apache-2.0
Author: Eduardo Ch. Colorado
E-mail: edxz7c@gmail.com
"""

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from torchvision import transforms
from pathlib import Path
# from fastai import 
from fastai.vision import open_image, load_learner
import io
import sys
import base64
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import importlib.util

# import cv2

path = Path(__file__).parent
models_path = path / 'models'
# build a dict with the avaible models, if an url isn't provided
# it's assumed that the model is already in the models folder 
avaible_models = {}
with open(path / "avaible_models.txt", "r") as f:
    for line in f:
        line = line.split()
        avaible_models[line[0]] = line[1]
        if(line[1] == 'none'): avaible_models[line[0]] = str(path / 'models' / line[0])
# what is the model we are going to use?
model_name = sys.argv[2]              
# is the name stored in the avaible models?
file_url = None 
pth_file = models_path / ("{}".format(model_name) + ".pth")
pkl_file = models_path / ("{}".format(model_name) + ".pkl")
if(model_name in avaible_models.keys() ):
    if(not pth_file.is_file() or not pkl_file.is_file()):
        file_url=avaible_models[model_name]
else:
    raise NotImplementedError("model name {} not implemented, please use add_model first to add your model".format(model_name))

# Default output
res = {"result": 0,
       "data": [], 
       "error": ''}

# create and set the starlette app
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

def model_loader():
    import importlib.util
    spec = importlib.util.spec_from_file_location("return_model", models_path / "{}.py".format(model_name))
    model_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_data)
    model = model_data.return_model()
    # load the data from the .pth to our current model 
    model.load_state_dict(torch.load(path/'models'/'{}.pth'.format(model_name), map_location='cpu')) 
    return model

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_model():
    if (pkl_file.is_file()):
        model = load_learner(models_path, model_name+".pkl")
        return model 
    elif(pth_file.is_file()):
        return model_loader()

    elif(avaible_models[model_name].split("/")[0]=="https:"):
        await download_file(file_url, models_path/model_name)
        try: # is a pickle file?
            model = load_learner(models_path, model_name)
            file_name = models_path/model_name
            file_name.replace(file_name.with_suffix('.pkl'))
            return model 
        except: # if isn't a pickle file it is a .pth file, right now the app onnly suport .pth files saved in pytorch
            file_name = models_path/model_name
            file_name.replace(file_name.with_suffix('.pth'))
            return model_loader()
    else:
        raise Exception("something goes wrong, maybe your url doesn't start with http:")


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_model())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/', methods=['GET'])
async def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['GET','POST'])
async def analyze(request):
    try:
        if request.method == 'POST':
            try:
                data = await request.body()
            except RuntimeError:
                data = "Data not available"
            data = str(data)    
            image_encoded = data.split(',')[1]
            image_bytes = io.BytesIO(base64.b64decode(image_encoded))
            if(pth_file.is_file()): 
                res['result'], res['data'] = predict_with_torch(image_bytes)
            elif(pkl_file.is_file()):    
                res['result'], res['data'] = predict_with_fastai(image_bytes)
            return JSONResponse(res)
    except Exception as e:
        # Return error data
        res['error'] = str(e)
        return JSONResponse(res)




def predict_with_fastai(image_bytes):
    print("inside 2")
    image = open_image(image_bytes, convert_mode='L')
    pred_class, pred_idx, probs = model.predict(image.resize(28))
    return 1, [float(num) for num in probs.numpy()] 

def predict_with_torch(image_bytes):
            image = Image.open(image_bytes).convert(mode='L')
            image.thumbnail((28,28))
            # resize_image = transforms.Compose([transforms.Resize((28,28))])
            # image = resize_image(image)
            # arr_image = np.array(image)
            # arr_image = cv2.cvtColor(arr_image, cv2.COLOR_RGB2GRAY)
            # tfm_image = transforms.Compose([transforms.ToTensor()])
            # arr_image = tfm_image(arr_image)
            arr_image = torch.tensor(np.array(image))
            arr_image = arr_image.type('torch.FloatTensor')[None, None, :, :]  
            # uncomment for debuggin (watch how the feeded image looks)
            # import matplotlib
            # matplotlib.use('TkAgg') 
            # from matplotlib import pyplot as plt
            # plt.imshow(np.squeeze(arr_image), cmap='gray')
            # plt.show()
            # Predict class
            #print(arr_image.shape)
            with torch.no_grad():
            # get sample scores
                model.eval() # for the capsule net, one of the outputs are probs not logits
                caps_output, reconstructions, probs, _ = model(arr_image)
            # Uncomment to observe the reconstructed image debugging
            # plt.imshow(np.squeeze(reconstructions.view(-1, 1, 28, 28)), cmap='gray')
            # plt.show()
            #logits = F.log_softmax(scores, dim=1)
            #probs = torch.exp(logits)
            #p, preds = probs.topk(10,dim=1)
            # Uncomment for debbuging
            # print("total prob ", probs.sum())
            return 1, [float(num) for num in probs.squeeze()]

    
if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
