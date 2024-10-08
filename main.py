from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch


app = Flask(__name__)

# Global variables for model and processor
model = None
processor = None
model_name = 'allenai/Molmo-7B-O-0924'

def load_model():
    global model, processor
    torch_dtype = torch.bfloat16

    # Load the processor and model
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map='cuda'
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

@app.route('/warm_up', methods=['GET'])
def warm_up():
    load_model()  # This should take a few minutes to load the model
    return jsonify({"status": "Model loaded successfully."})

@app.route('/generate_text', methods=['POST'])
def generate_text():
    global model, processor

    if model is None or processor is None:
        return jsonify({"error": "Model not loaded. Please call /warm_up first."}), 400

    # Get the image and prompt from the request
    try:
        prompt = request.form['prompt']
        image_file = request.files['image']

        # Process the image
        image = Image.open(image_file)

        # Process the inputs
        inputs = processor.process(
            images=[image],
            text=prompt
        )
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # Generate the output from the model
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


        result = {
            "generated_text": generated_text
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
