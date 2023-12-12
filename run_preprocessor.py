import argparse
from collections import defaultdict
import csv
import os
from uuid import uuid4 as uuid

from src_preprocessor.casetext import extract_text_from_casetext
from src_preprocessor.scotus import download_and_parse_pdf
from src_preprocessor.process_text import find_and_save_strings

RAW_CASEFILE_FOLDER = "casefiles_raw"
DATASET_CSV = "dataset.csv"
CASETEXT_URL_PREFIX = "https://casetext.com/case/"

CSV_TITLE = "dictionary.csv"
QUOTES_CSV_TITLE = 'quotes.csv'
ID_COLUMN = "file_id"
CASE_CITATION = "citation"
CASE_CITATION_FULL = "citation_full"
CASE_TITLE = "case_title"
SOURCE_URL = "source_url"
CASE_CSV_COLUMNS = [ID_COLUMN, SOURCE_URL, CASE_CITATION, CASE_CITATION_FULL, CASE_TITLE]
DATASET_FIELD_NAMES = [ID_COLUMN, 'para_n', 'word_n', 'word_c', 'sentence']

TYPE_CASETEXT = "casetext"
TYPE_SCOTUS = "scotus"
TYPE_NONE = "none"

RAW_WRITING_FOLDER = "writings_raw"
PDF_ID_COLUMN = "pdf_id"
TEXT_ID_COLUMN = "text_id"
WRITING_CSV_COLUMNS = [PDF_ID_COLUMN, TEXT_ID_COLUMN, SOURCE_URL]

def add_to_csv(csv_path, field_names, new_entry):
	if os.path.isfile(csv_path):
		with open(csv_path, 'a', newline='') as file:
			writer = csv.DictWriter(file, fieldnames=field_names)
			writer.writerow(new_entry)
	else:
		with open(csv_path, 'w', newline='') as file:
			writer = csv.DictWriter(file, fieldnames=field_names)
			writer.writeheader()
			writer.writerow(new_entry)

def build_citation_lookup():
	out = {}
	cases_csv_path = os.path.join(RAW_CASEFILE_FOLDER, CSV_TITLE)
	with open(cases_csv_path, 'r', newline='') as file:
		reader = csv.DictReader(file)
		for row in reader:
			out[row[CASE_CITATION]] = (os.path.join(RAW_CASEFILE_FOLDER, row[ID_COLUMN]), row[ID_COLUMN])

	return out

def process(url, type):
	if type == TYPE_NONE:
		if "casetext" in url:
			type = TYPE_CASETEXT
		elif "supremecourt" in url:
			type = TYPE_SCOTUS
		else:
			type = TYPE_SCOTUS #See what happens
			#print("No type specified and unclear URL provided")		

	if type == TYPE_CASETEXT:
		if url[:4] != "http" and url[:8] != "casetext":
			url = CASETEXT_URL_PREFIX + url

		file_id = str(uuid())
		destination_file = os.path.join(RAW_CASEFILE_FOLDER, file_id)

		metadata = extract_text_from_casetext(url, destination_file)
		if metadata["failed"] == False:
			case_title = metadata["title"]
			citation = metadata["citation"]
			citation_full = metadata["citation_full"]

			#### Add entry to CSV ####
			csv_path = os.path.join(RAW_CASEFILE_FOLDER, CSV_TITLE)
			new_entry = {ID_COLUMN: file_id, SOURCE_URL: url, CASE_CITATION: citation,
						CASE_CITATION_FULL: citation_full, CASE_TITLE: case_title}
			
			add_to_csv(csv_path, CASE_CSV_COLUMNS, new_entry)

		text = metadata["text"]
		while (i := text.find('*')) >= 0:
			while i+1 < len(text) and text[i+1].isdigit():
				text = text[:i+1] + text[i+2:]
			text = text[:i] + text[i+1:]
		text = text.replace('  ', ' ')
		text = text.replace('  ', ' ')
		text = text.replace('  ', ' ')

		if args.search_quotes:
			find_and_save_strings(text, os.path.join(RAW_CASEFILE_FOLDER, QUOTES_CSV_TITLE))

	elif type == TYPE_SCOTUS:
		pdf_id = str(uuid())
		pdf_destination = os.path.join(RAW_WRITING_FOLDER, pdf_id + '.pdf')
		
		text_id = str(uuid())
		text_destination = os.path.join(RAW_WRITING_FOLDER, text_id + '.txt')

		text = download_and_parse_pdf(url, pdf_destination, text_destination)
		if not text:
			print(f"Error accessing {url}")
			return False

		csv_path = os.path.join(RAW_WRITING_FOLDER, CSV_TITLE)
		new_entry = {PDF_ID_COLUMN: pdf_id, TEXT_ID_COLUMN:text_id, SOURCE_URL:url}

		add_to_csv(csv_path, WRITING_CSV_COLUMNS, new_entry)

		if args.search_quotes:
			find_and_save_strings(text, os.path.join(RAW_CASEFILE_FOLDER, QUOTES_CSV_TITLE))

def get_numbers_for_quote(text, quote):
	paragraphs = text.split('\n\n')
	para_n, word_n, word_c = -1, -1, -1
	for i in range(len(paragraphs)):
		if quote in paragraphs[i]:
			para_n = i
			paragraph = paragraphs[i]
			break
	i = paragraph.find(quote)
	pretext = paragraph[:i-1]
	word_n = len(pretext.split())
	word_c = len(quote.split())

	return para_n, word_n, word_c


def audit_quotes():
	quotes_csv_path = os.path.join(RAW_CASEFILE_FOLDER, QUOTES_CSV_TITLE)

	citation_lookup = build_citation_lookup()
	count = 0
	true_count = 0
	to_add = defaultdict(int)
	with open(quotes_csv_path, 'r', newline='') as file:
		reader = csv.DictReader(file)
		for row in reader:
			if row['citation'] in citation_lookup:
				count += 1
				with open(citation_lookup[row['citation']][0], 'r', newline='') as file2:
					text = file2.read()
				if row["quote"] in text:
					true_count += 1
					para_n, word_n, word_c = get_numbers_for_quote(text, row["quote"])

					print(citation_lookup[row['citation']][1], para_n, word_n, word_c, row['sentence'])
					new_entry = {
						ID_COLUMN: citation_lookup[row['citation']][1],
				  		'para_n': para_n, 'word_n': word_n, 'word_c':word_c,
						'sentence': row['sentence']
					}
					add_to_csv(DATASET_CSV, DATASET_FIELD_NAMES, new_entry)
			else:
				citation = ' '.join([row['title'], row['citation']])
				to_add[citation] += 1
	
	to_share = sorted(to_add.items(), key=lambda x: x[1], reverse=True)[:5]

	print(f"So far, your dataset contains {count} valid samples")
	print(f"But your true count is {true_count}")
	print("")
	print("Here are some cases you could download that would increase that count:")
	for item in to_share:
		print(item)


parser = argparse.ArgumentParser(description="Downloads decisions from Casetext")

parser.add_argument(
    "--url", help="url to download"
)

parser.add_argument(
    "--type", default=TYPE_NONE, choices=[TYPE_CASETEXT, TYPE_SCOTUS], help="type of download"
)

parser.add_argument(
    "--source-file", help="file with list of urls"
)

parser.add_argument('--search-quotes', action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

# Get the full URL
url = args.url

type = args.type

if args.source_file:
	urls_visited = set()
	for csv_path in [os.path.join(RAW_CASEFILE_FOLDER, CSV_TITLE), os.path.join(RAW_WRITING_FOLDER, CSV_TITLE)]:
		try:
			with open(csv_path, 'r', newline='') as file:
				reader = csv.DictReader(file)
				for row in reader:
					urls_visited.add(row[SOURCE_URL])
		except:
			# if it doesn't exist, do nothing
			pass
	with open(args.source_file, 'r') as file:
		for line in file:
			if (url := line.strip()) and url[0] != '#':
				if url not in urls_visited:
					urls_visited.add(url)
					process(url, type)
	audit_quotes()
elif not url:
	print('--url or --source-file is required')
else:
	process(url, type)
