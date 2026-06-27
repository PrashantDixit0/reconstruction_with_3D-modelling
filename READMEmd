## FURNITURE COMPONENT SWAP — Documentation                  
                                                                              
  DESCRIPTION                                                                 
  -----------                                                                 
  Swaps a furniture component (e.g. armrest, leg, backrest) from a source     
  image onto a target image using:                                            
    • LLaVA  — Vision-language model for furniture identification             
    • SAM    — Segment Anything Model for component segmentation              
    • SD     — Stable Diffusion (always on) for seamless inpainting           
                                                                              
  INSTALLATION                                                                
  ------------                                                                
1. Python >= 3.10                                                           
                                                                            
2. Install Ollama and pull LLaVA:                                           
       ```
       https://ollama.com/download                                            
       ollama pull llava      
       ```                                                
                                                                              
3. Install Python dependencies:                                            
```    
    pip install ollama opencv-python pillow numpy torch torchvision 

    pip install git+https://github.com/facebookresearch/segment-anything  

    pip install requests
```
                                                                              
  4. Download SAM checkpoint (ViT-H, ~2.5 GB):                               
```
     wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    
```

Place it in the same directory as this script, or pass --sam-checkpoint
                                                                              
  5. Stable Diffusion inpainting (required):                                  
       Install AUTOMATIC1111 
```
WebUI:                                           
    https://github.com/AUTOMATIC1111/stable-diffusion-webui             
Launch it with: webui.sh --api                                         
Default URL: http://127.0.0.1:7860

```                                  
                                                                              
  USAGE                                                                       
  -----                                                                       
  Basic usage (SD inpainting always enabled):                                 
```
python furniture_swap.py \                                                
--source chair_with_arm.png \                                           
--target chair_without_arm.png \                                        
--component armrest                                       
``` 
                                                                              
  With custom SD URL and strength:                                            
```
python furniture_swap.py \                                                
    --source chair_with_arm.png \                                           
    --target chair_without_arm.png \                                        
    --component armrest \                                                   
    --sd-url http://192.168.1.10:7860 \                                     
    --denoising-strength 0.65
```                                             
                                                                              
  With custom SAM checkpoint and output path:                                 
```
python furniture_swap.py \                                                
--source sofa.png \                                                     
--target plain_sofa.png \                                               
--component leg \                                                       
--sam-checkpoint /models/sam_vit_h_4b8939.pth \                        
--output result.png \                                                   
--denoising-strength 0.65
```                                           
                                                                              
CLI  ARGUMENTS                                                         
-----------  

| Argument               | Description                                                                                                                                                 | Default                                    | Example / Allowed Values                        |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------- |
| `--source`             | Path to the source image containing the component to copy.                                                                                                  | —                                          | `source.png`                                    |
| `--target`             | Path to the target image that will receive the copied component.                                                                                            | —                                          | `target.png`                                    |
| `--component`          | Name of the furniture component to swap.                                                                                                                    | `armrest`                                  | `armrest`, `leg`, `seat`, `backrest`, `cushion` |
| `--output`             | Path to save the generated output image.                                                                                                                    | `swapped_furniture.png`                    | `output.png`                                    |
| `--sd-url`             | URL of the Stable Diffusion WebUI API endpoint.                                                                                                             | `http://127.0.0.1:7860`                    | `http://localhost:7860`                         |
| `--denoising-strength` | Denoising strength for Stable Diffusion (`0.0–1.0`). Lower values preserve more of the original image, while higher values allow more aggressive redrawing. | `0.75`                                     | `0.0` – `1.0`                                   |
| `--sd-steps`           | Number of Stable Diffusion sampling steps. Higher values may improve quality at the cost of longer processing time.                                         | `30`                                       | `20`, `30`, `50`                                |
| `--sam-checkpoint`     | Path to the Segment Anything Model (SAM) checkpoint (`.pth`) file.                                                                                          | `sam_vit_h_4b8939.pth` (current directory) | `models/sam_vit_h_4b8939.pth`                   |
| `--sam-model-type`     | Type of SAM model to use.                                                                                                                                   | `vit_h`                                    | `vit_h`, `vit_l`, `vit_b`                       |
                          
                                                                              

3D Modelling                                                              
  -----------  

Once swapped is ready for performing 3D modelling on it.
Utilize the Stable Diffusion's TripoSR [Gradio Space](https://huggingface.co/spaces/stabilityai/TripoSR) and create 3D model of the Output image.