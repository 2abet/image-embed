import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import gradio as gr
import spaces

processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)

def ImgEmbed(image):
    """
    Generate normalized embedding vector for the uploaded image.

    Args:
        image (PIL.Image.Image or np.ndarray): Input image uploaded by the user.

    Returns:
        list[float]: A normalized image embedding vector representing the input image.
    """
    print(image);
    inputs = processor(image, return_tensors="pt")

    img_emb = vision_model(**inputs).last_hidden_state
    img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)

    return img_embeddings[0].tolist();



with gr.Blocks() as demo:      
       img = gr.Image();
       out = gr.Text();
       
       btn = gr.Button("Get Embeddings")
       btn.click(ImgEmbed, [img], [out])
       
       
if __name__ == "__main__":
    demo.launch(mcp_server=True)