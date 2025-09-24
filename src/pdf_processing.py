import os
import glob
import fitz
import logging

def process_input_source(source):
	pdf_files = []
	if isinstance(source, str):
		if os.path.isdir(source):
			pdf_files.extend(glob.glob(os.path.join(source, "*.pdf")))
		elif os.path.isfile(source) and source.lower().endswith('.pdf'):
			pdf_files.append(source)
	return pdf_files

def process_multiple_pdfs(pdf_paths, logger=None):
	all_chunks = []
	all_page_numbers = []
	all_sources = []

	for pdf_path in pdf_paths:
		try:
			doc = fitz.open(pdf_path)
			pdf_name = os.path.basename(pdf_path)
			for page_num, page in enumerate(doc, 1):
				text = page.get_text()
				chunks = text.split('\n\n')
				for chunk in chunks:
					if chunk.strip():
						all_chunks.append(chunk.strip())
						all_page_numbers.append(page_num)
						all_sources.append(pdf_name)
			doc.close()
			if logger:
				logger.info(f"Processed {pdf_name}: {len(chunks)} chunks extracted")
		except Exception as e:
			if logger:
				logger.error(f"Error processing {pdf_path}: {str(e)}")
	return all_chunks, all_page_numbers, all_sources
