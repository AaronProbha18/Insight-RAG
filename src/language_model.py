import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_phi2_model(model_id="microsoft/phi-2"):
	tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		device_map="auto",
		torch_dtype=torch.float16,
		trust_remote_code=True
	)
	return tokenizer, model

def generate_response(tokenizer, model, prompt, max_new_tokens=500, logger=None):
	try:
		inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
		outputs = model.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			temperature=0.7,
			do_sample=True,
			top_p=0.95,
			pad_token_id=tokenizer.eos_token_id
		)
		response = tokenizer.decode(outputs[0], skip_special_tokens=True)
		cleaned_response = response
		if "Response:" in response:
			cleaned_response = response.split("Response:")[-1].strip()
		elif "Answer:" in response:
			cleaned_response = response.split("Answer:")[-1].strip()
		return cleaned_response
	except Exception as e:
		if logger:
			logger.error(f"Error in generate_response: {str(e)}")
		raise e
