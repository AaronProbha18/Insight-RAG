import os
import sys
import logging
import datetime
import nest_asyncio
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pyngrok import ngrok

from pdf_processing import process_multiple_pdfs
from embeddings import get_embedding_model, generate_embeddings
from vector_store import SimpleVectorStore
from language_model import load_phi2_model, generate_response

# Logging setup
logs_dir = Path("rag_logs")
logs_dir.mkdir(exist_ok=True)
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = logs_dir / f"rag_session_{session_timestamp}.log"

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	handlers=[
		logging.StreamHandler(sys.stdout),
		logging.FileHandler(log_filename)
	]
)
logger = logging.getLogger(__name__)
session_history = []

def save_session_summary():
	try:
		summary_filename = logs_dir / f"session_summary_{session_timestamp}.txt"
		with open(summary_filename, 'w', encoding='utf-8') as f:
			f.write(f"RAG Session Summary - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			f.write("="*80 + "\n\n")
			if session_history:
				for i, interaction in enumerate(session_history, 1):
					f.write(f"Interaction {i}:\n")
					f.write("-"*40 + "\n")
					f.write(f"Query: {interaction['query']}\n")
					f.write(f"Timestamp: {interaction['timestamp']}\n")
					f.write(f"Retrieved Pages: {interaction['citations']}\n")
					f.write(f"Response: {interaction['response']}\n")
					f.write("\n" + "="*40 + "\n\n")
			else:
				f.write("No queries were made during this session.\n")
		logger.info(f"Session summary saved to {summary_filename}")
	except Exception as e:
		logger.error(f"Error saving session summary: {e}")

# --- HTML Templates (truncated for brevity, copy from notebook as needed) ---
INITIAL_TEMPLATE = """<html><!-- ... --></html>"""
HTML_TEMPLATE = """<html><!-- ... --></html>"""

# --- Utility Functions for RAG Pipeline ---
import re
def convert_markdown_to_html(text):
	text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
	text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', text)
	return text

def remove_duplicate_lines(text):
	lines = text.split("\n")
	seen = set()
	new_lines = []
	for line in lines:
		stripped_line = line.strip()
		if stripped_line and stripped_line not in seen:
			seen.add(stripped_line)
			new_lines.append(line)
	return "\n".join(new_lines)

def remove_repeated_questions(text, question):
	pattern = re.compile(re.escape(question), re.IGNORECASE)
	occurrences = pattern.findall(text)
	if len(occurrences) > 1:
		first_occurrence = True
		def replacer(match):
			nonlocal first_occurrence
			if first_occurrence:
				first_occurrence = False
				return match.group(0)
			else:
				return ""
		text = pattern.sub(replacer, text)
	return text

def format_answer(question, answer, citations):
	answer = convert_markdown_to_html(answer)
	def structure_content(text):
		sections = text.split('\n\n')
		main_content = sections[0]
		lists = []
		for section in sections[1:]:
			if any(line.strip().startswith(('•', '-', '*', '1.', '2.', '3.')) for line in section.split('\n')):
				lists.append(section)
		return main_content, lists
	main_content, additional_sections = structure_content(answer)
	list_html = ""
	if additional_sections:
		list_html = "<div class='additional-content'>"
		for section in additional_sections:
			items = [line.strip().lstrip('•-*123456789. ') for line in section.split('\n') if line.strip()]
			list_html += "<ul class='content-list'>"
			for item in items:
				list_html += f"<li>{convert_markdown_to_html(item)}</li>"
			list_html += "</ul>"
		list_html += "</div>"
	citations_html = " ".join(
		f'<span class="citation-tag">{citation}</span>'
		for citation in citations
	)
	return f"""
	<div class="answer-message">
		<div class="answer-header">
			<div class="question-text">Q: {question}</div>
		</div>
		<div class="answer-content">
			<div class="main-content">{main_content}</div>
			{list_html}
		</div>
		<div class="answer-footer">
			<div class="citations-section">
				<span class="citations-label">Sources:</span>
				<div class="citations-container">{citations_html}</div>
			</div>
		</div>
	</div>
	"""

def rag_pipeline(query, vector_store, embedding_model, tokenizer, model, max_context_length=5000):
	try:
		query_embedding = embedding_model.encode([query])[0]
		relevant_texts_with_info = vector_store.search(query_embedding)
		citations = []
		context_parts = []
		total_length = 0
		context_set = set()
		for text, page, source in relevant_texts_with_info:
			cleaned_text = text.strip()
			if cleaned_text not in context_set:
				context_set.add(cleaned_text)
				citations.append(f"{source} (Page {page})")
				if len(cleaned_text) > max_context_length:
					cleaned_text = cleaned_text[:max_context_length] + "..."
				if total_length + len(cleaned_text) <= max_context_length:
					context_parts.append(cleaned_text)
					total_length += len(cleaned_text)
				else:
					break
		context = " ".join(context_parts)
		prompt = f"""You are a chatbot which answers question(s) to the point. The context will contain the relevant information retrieved from a vector store based on vector search. The question will be question asked by the user from a web chatbot. If the information provided is not sufficient to answer the question accurately, state that you don't have enough information to provide a reliable answer. Do not refer the context in the answer. Do not give additional references to the information as this will be given as citations separately. Do not have any prefix to the answer like 'according to the context' or something. Start with the answer. Try to answer the question using the context and the question. Only when asked for troubleshooting also find out the troubleshooting steps for the same if present in the context. The output should only contain the answer and nothing else. If there is no context provided then please repond \"Sorry. I don't find information for the question.\" And dont create answer on your own.\n\nQuestion: {query}\n\nAnswer:"""
		response = generate_response(tokenizer, model, prompt)
		response = remove_duplicate_lines(response)
		response = remove_repeated_questions(response, query)
		session_history.append({
			'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
			'query': query,
			'citations': citations,
			'response': response
		})
		return format_answer(query, response, citations)
	except Exception as e:
		return f"""
		<div class=\"error\">I apologize, but I encountered an error while processing your query. Please try asking a more specific or shorter question.</div>"""

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)
vector_store = None
embedding_model = get_embedding_model()
tokenizer, model = load_phi2_model()

@app.after_request
def after_request(response):
	response.headers.add('Access-Control-Allow-Origin', '*')
	response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
	response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
	return response

@app.route('/', methods=['GET'])
def index():
	return render_template_string(INITIAL_TEMPLATE)

@app.route('/chat', methods=['GET'])
def chat():
	if vector_store is None:
		return redirect('/')
	return render_template_string(HTML_TEMPLATE)

@app.route('/setup', methods=['POST'])
def setup():
	try:
		method = request.form.get('method')
		pdf_files = []
		if 'file' not in request.files:
			return jsonify({'success': False, 'error': 'No files uploaded'})
		files = request.files.getlist('file')
		if not files or files[0].filename == '':
			return jsonify({'success': False, 'error': 'No files selected'})
		if not os.path.exists(uploads_dir):
			os.makedirs(uploads_dir)
		upload_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		current_upload_dir = uploads_dir / f"upload_{upload_timestamp}"
		current_upload_dir.mkdir(exist_ok=True)
		for file in files:
			if file and file.filename.lower().endswith('.pdf'):
				if method == '1':
					filename = secure_filename(file.filename)
					filepath = str(current_upload_dir / filename)
				else:
					relative_path = Path(file.filename)
					filepath = str(current_upload_dir / relative_path.name)
				os.makedirs(os.path.dirname(filepath), exist_ok=True)
				file.save(filepath)
				pdf_files.append(filepath)
				logger.info(f"Saved file: {file.filename}")
		if not pdf_files:
			return jsonify({'success': False, 'error': 'No valid PDF files found'})
		logger.info(f"Processing {len(pdf_files)} PDF files...")
		global vector_store
		text_chunks, page_numbers, sources = process_multiple_pdfs(pdf_files, logger=logger)
		if not text_chunks:
			return jsonify({'success': False, 'error': 'No text content extracted from PDFs'})
		embeddings = generate_embeddings(embedding_model, text_chunks)
		vector_store = SimpleVectorStore(embeddings, text_chunks, page_numbers, sources)
		logger.info("Vector store created successfully")
		return jsonify({'success': True})
	except Exception as e:
		logger.error(f"Error in setup: {str(e)}")
		return jsonify({'success': False, 'error': str(e)})

@app.route('/', methods=['POST'])
def process_query():
	if not vector_store:
		return jsonify({'error': 'No documents loaded. Please setup the system first.'})
	query = request.form.get('query')
	if not query:
		return jsonify({'error': 'No query provided'})
	try:
		response = rag_pipeline(query, vector_store, embedding_model, tokenizer, model)
		return jsonify({'response': response})
	except Exception as e:
		return jsonify({
			'error': 'An error occurred while processing your query',
			'details': str(e)
		})

@app.route('/shutdown', methods=['POST'])
def shutdown():
	save_session_summary()
	return jsonify({'success': True})

def run_flask_with_public_url():
	try:
		nest_asyncio.apply()
		ngrok.kill()
		public_url = ngrok.connect(5000)
		print("\n" + "="*50)
		print("🚀 RAG System is Live!")
		print("="*50)
		print(f"Main URL: {public_url}")
		print("="*50)
		app.run(
			host='0.0.0.0',
			port=5000,
			debug=False,
			use_reloader=False
		)
	except Exception as e:
		print(f"Error starting server: {str(e)}")
		raise e

if __name__ == "__main__":
	run_flask_with_public_url()
