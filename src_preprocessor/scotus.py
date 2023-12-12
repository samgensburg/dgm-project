import requests

from pypdf import PdfReader
import fitz

def download_pdf(url, filepath):
	response = requests.get(url)
	if response.status_code == 200:
		with open(filepath, 'wb') as file:
			file.write(response.content)
		return True
	else:
		print("Failed to retrieve the webpage. Status code:", response.status_code)
		return False


def extract_text_from_pdf_pypdf(pdf_path):
	with open(pdf_path, 'rb') as file:
		reader = PdfReader(file)
		text = ''
		for page in reader.pages:
			text += page.extract_text() + "\n"
		return text.replace('“','"').replace('”','"')


def extract_text_from_pdf(pdf_path):
	doc = fitz.open(pdf_path)
	text = ''
	for page in doc:
		text += page.get_text() + "\n"
	return text.replace('“','"').replace('”','"')


def download_and_parse_pdf(url, pdf_path, text_path):
	if not download_pdf(url, pdf_path):
		return False
	text = extract_text_from_pdf(pdf_path)
	with open(text_path, 'w') as file:
		file.write(text)

	return text
