import re
import requests

from bs4 import BeautifulSoup, NavigableString

def get_text_recursive(element):
	if isinstance(element, NavigableString):
		return element.strip()
	elif element.name in ["i"]:
		# Preserving italics in case I want to treat them as special terms
		return f"*{element.get_text(strip=True)}*"
	elif element.name in ["a", "b"]:
		return f"{element.get_text(strip=True)}"
	elif element.name in ["span"]:
		if not element.get("page-number"):
			return ''
		return f"*{element.get('page-number')}"
	elif element.name in ["p", "h3", "blockquote"]:
		# p is normal text, h3 is for headings, blockquote is for block quotes
		return ' '.join(get_text_recursive(child) for child in element.children)
	elif element.name in ["div"]:
		return '' # To my knowledge, these are all footnotes, which I have decided to skip.
	else:
		print(f"Found unexpected element: {element.name}")
		return ''


def get_metadata(response):
	out = dict()
	soup = BeautifulSoup(response.text, "html.parser")

	header = soup.find_all("h1") # This might not work, but I'm trying for now
	out['title'] = header[0].get_text(strip=True)

	citation = soup.find_all(class_="citation") # This might not work, but I'm trying for now
	citation_text = citation[0].get_text(strip=True)
	out['citation_full'] = citation_text
	match = re.match(r"(\d+ .+ \d+) (\((?:.* )?\d{4}\))", citation_text)
	if not match:
		print(f"match failed for {citation_text}")
		return None

	out['citation'] = match.group(1)
	out['citation_parenthetical'] = match.group(2)

	return out


def extract_text_from_casetext(url, destination):
	response = requests.get(url)

	if response.status_code == 200:
		soup = BeautifulSoup(response.text, "html.parser")
        
		main_content = soup.find_all(class_="decision opinion")

		main_content_lengths = [len(e.get_text()) for e in main_content]
		main_content = main_content[main_content_lengths.index(max(main_content_lengths))]

		children = main_content.findChildren(recursive=False)
		paragraph_text_strings = []
		for element in children:
			element_text = get_text_recursive(element)
			paragraph_text_strings.append(element_text)

		paragraph_text_strings = [s.replace('\n', ' ') for s in paragraph_text_strings if len(s.strip())]

		main_text = '\n\n'.join(paragraph_text_strings)

		# Save the text to a file
		with open(destination, 'w') as file:
			file.write(main_text)

		metadata = get_metadata(response)
		if metadata != None:
			metadata["text"] = main_text
			metadata["failed"] = False
		else:
			metadata = {"text": main_text, "failed": True}
		return metadata
	else:
		print("Failed to retrieve the webpage. Status code:", response.status_code)
