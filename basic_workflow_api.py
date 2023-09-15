import json
from urllib import request, parse
import random

# ======================================================================
# This function sends a prompt workflow to the specified URL 
# (http://127.0.0.1:8188/prompt) and queues it on the ComfyUI server
# running at that address.
def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)    
# ======================================================================

# read workflow api data from file and convert it into dictionary 
# assign to var prompt_workflow
prompt_workflow = json.load(open('workflow_api.json'))

# create a list of prompts
prompt_list = []
prompt_list.append("photo of a man sitting in a cafe")
prompt_list.append("photo of a woman standing in the middle of a busy street")
prompt_list.append("drawing of a cat sitting in a tree")
prompt_list.append("beautiful scenery nature glass bottle landscape, purple galaxy bottle")

# give some easy-to-remember names to the nodes
chkpoint_loader_node = prompt_workflow["4"]
prompt_pos_node = prompt_workflow["6"]
empty_latent_img_node = prompt_workflow["5"]
ksampler_node = prompt_workflow["3"]
save_image_node = prompt_workflow["9"]

# load the checkpoint that we want. 
chkpoint_loader_node["inputs"]["ckpt_name"] = "SD1-5/sd_v1-5_vae.ckpt"

# set image dimensions and batch size in EmptyLatentImage node
empty_latent_img_node["inputs"]["width"] = 512
empty_latent_img_node["inputs"]["height"] = 640
# each prompt will produce a batch of 4 images
empty_latent_img_node["inputs"]["batch_size"] = 4

# for every prompt in prompt_list...
for index, prompt in enumerate(prompt_list):

  # set the text prompt for positive CLIPTextEncode node
  prompt_pos_node["inputs"]["text"] = prompt

  # set a random seed in KSampler node 
  ksampler_node["inputs"]["seed"] = random.randint(1, 18446744073709551614)

  # if it is the last prompt
  if index == 3:
    # set latent image height to 768
    empty_latent_img_node["inputs"]["height"] = 768

  # set filename prefix to be the same as prompt
  # (truncate to first 100 chars if necessary)
  fileprefix = prompt
  if len(fileprefix) > 100:
    fileprefix = fileprefix[:100]

  save_image_node["inputs"]["filename_prefix"] = fileprefix

  # everything set, add entire workflow to queue.
  queue_prompt(prompt_workflow)
