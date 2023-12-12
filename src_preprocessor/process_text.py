from collections import defaultdict
import csv
import os
import re

import nltk
from nltk.tokenize import sent_tokenize

BLUEBOOK_T6_PERIOD_ENDINGS = [
	'Acad.',
	'Acct.',
	'Admin.',
	'Advert.',
	'Advoc.',
	'Aff.',
	'Afr.',
	'Agric.',
	'All.',
	'Amend.',
	'Am.',
	'Anc.',
	'Ann.',
	'Arb.',
	'A.I.',
	'Assoc.',
	'Atl.',
	'Auth.',
	'Auto.',
	'Ave.',
	'Bankr.',
	'Behav.',
	'Bd.',
	'Brit.',
	'Broad.',
	'Bros.',
	'Bhd.',
	'Bldg.',
	'Bull.',
	'Bus.',
	'Cap.',
	'Cas.',
	'Cath.',
	'Ctr.',
	'Cent.',
	'Chem.',
	'Chron.',
	'Cir.',
	'Civ.',
	'Coll.',
	'Com.',
	'Comm.',
	'Cmty.',
	'Co.',
	'Compar.',
	'Comp.',
	'Comput.',
	'Condo.',
	'Conf.',
	'Cong.',
	'Consol.',
	'Const.',
	'Constr.',
 	'Contemp.',
 	'Cont.',
 	'Conv.',
 	'Coop.',
 	'Corp.',
 	'Corr.',
 	'Cosm.',
 	'Couns.',
 	'Cnty.',
 	'Ct.',
 	'Crim.',
 	'Def.',
 	'Delinq.',
 	'Det.',
 	'Dev.',
 	'Digit.',
 	'Dipl.',
 	'Dir.',
 	'Disc.',
 	'Disp.',
 	'Distrib.',
 	'Dist.',
 	'Div.',
 	'Dr.',
 	'Econ.',
 	'Ed.',
 	'Educ.',
 	'Elec.',
 	'Emp.',
 	'Enter.',
 	'Ent.',
 	'Equip.',
 	'Est.',
 	'Eur.',
 	'Exch.',
 	'Exec.',
 	'Expl.',
 	'Exp.',
 	'Fac.',
 	'Fam.',
 	'Fed.',
 	'Fid.',
 	'Fin.',
 	'Gen.',
 	'Glob.',
 	'Grp.',
 	'Guar.',
 	'Hist.',
 	'Hosp.',
 	'Hous.',
 	'Immigr.',
 	'Imp.',
	'Inc.',
 	'Indem.',
 	'Indep.',
 	'Indus.',
 	'Ineq.',
 	'Inj.',
 	'Inst.',
 	'Ins.',
 	'Intell.',
 	'Interdisc.',
 	'Inv.',
 	'Jud.',
 	'Jurid.',
 	'Juris.',
 	'Just.',
 	'Juv.',
 	'Lab.',
 	'Legis.',
 	'Liab.',
 	'Libr.',
 	'Ltd.',
 	'Litig.',
 	'Loc.',
 	'Mach.',
 	'Mag.',
 	'Maint.',
 	'Mgmt.',
 	'Mfr.',
 	'Mfg.',
 	'Mar.',
 	'Mkt.',
 	'Mktg.',
 	'Matrim.',
 	'Mech.',
 	'Med.',
 	'Merch.',
 	'Metro.',
 	'Mil.',
 	'Min.',
 	'Mod.',
 	'Mortg.',
 	'Mun.',
 	'Mut.',
 	'Nat.',
 	'Negl.',
 	'Negot.',
 	'Newsl.',
 	'Ne.',
 	'Nw.',
 	'No.',
 	'Off.',
 	'Op.',
 	'Ord.',
 	'Org.',
 	'Pac.',
 	'Par.',
 	'Pat.',
 	'Pers.',
 	'Persp.',
 	'Pharm.',
 	'Phil.',
 	'Plan.',
 	'Pol.',
 	'Prac.',
 	'Pres.',
 	'Priv.',
 	'Prob.',
 	'Probs.',
	'Proc.',
	'Prod.',
 	'Pro.',
	'Prop.',
	'Prot.',
	'Psych.',
 	'Pub.',
 	'Ry.',
 	'Rec.',
 	'Ref.',
 	'Refin.',
 	'Reg.',
 	'Regul.',
 	'Rehab.',
 	'Rel.',
 	'Rep.',
 	'Reprod.',
 	'Rsch.',
 	'Rsrv.',
	'Resol.',
 	'Res.',
 	'Resp.',
 	'Rest.',
 	'Ret.',
 	'Rev.',
 	'Rts.',
	'Rd.',
	'Sav.',
 	'Sch.',
	'Sci.',
	'Scot.',
	'Sec.',
	'Serv.',
	'Soc.',
	'Socio.',
	'Solic.',
	'Sol.',
 	'Se.',
 	'Sw.',
	'Stat.',
 	'Stud.',
	'Subcomm.',
	'Sur.',
	'Surv.',
	'Symp.',
	'Sys.',
	'Tchr.',
	'Tech.',
	'Telecomm.',
	'Tel.',
	'Temp.',
	'Twp.',
	'Transcon.',
	'Transp.',
	'Trib.',
	'Tr.',
	'Tpk.',
	'Unif.',
	'Univ.',
	'Urb.',
	'Util.',
	'Vill.',
	'Wk.',
	'Wkly.'
]

#nltk.download('punkt')

FULL_CITATION_REGEX = r'^([a-zA-Z \.\'\,]+ v\. [a-zA-Z \.\'\,]+),? (\d+ [a-zA-z0-9\.]+ \d+), \d+ \([^\)]*\d{4}\)\.$'
CITATION_WITHOUT_NAME_REGEX = r'^(\d+ [a-zA-z0-9\.]+ \d+), \d+ \([^\)]*\d{4}\)\.$'
SHORT_CITATION_REGEX = r'^([a-zA-Z \.\'\,]+), (\d+ [a-zA-z0-9\.]+) at (\d+)$'

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

def find_and_save_strings(text, path):
	text = clean_text(text)
	sentences = sent_tokenize(text)

	for i in range(len(sentences)):
		while i < len(sentences) - 1:
			if re.match(r'^at \d', sentences[i+1]):
				sentences = sentences_merge(sentences, i)
			elif sentences[i+1].strip() == '.':
				sentences = sentences_merge(sentences, i)
			elif sentences[i][-5:] == ' Cir.':
				sentences = sentences_merge(sentences, i)
			elif sentences[i+1][:3] == 'v. ':
				sentences = sentences_merge(sentences, i)
			elif ends_with_t6(sentences[i]):
				sentences = sentences_merge(sentences, i)
			elif has_one_quote(sentences[i]) and has_one_quote(sentences[i+1]):
				sentences = sentences_merge(sentences, i)
			else:
				break

	known_cases = defaultdict(set)
	for i in range(1, len(sentences)):
		sentence = sentences[i]
		previous = sentences[i-1]
		if is_plain_cite_string(previous):
			reporter = reporter_from_full_citation(previous)
			known_cases[reporter].add(previous)
		if is_plain_cite_string(sentence) and contains_usable_quote(previous):
			add_record(previous, sentence, path)
		if is_cite_string_without_name(sentence) and contains_usable_quote(previous):
			reporter = reporter_from_citation_without_name(sentence)
			for known_cite in known_cases[reporter]:
				if sentence in known_cite:
					sentence = title + ', ' + sentence
			add_record(previous, sentence, path)
		if is_short_cite(sentence) and contains_usable_quote(previous):
			reporter, title, pincite = reporter_title_pincite_from_short_cite(sentence)
			for known_cite in known_cases[reporter]:
				if title in known_cite:
					match = re.match(FULL_CITATION_REGEX, known_cite)
					sentence = match.group(1) + ', ' + match.group(2) + ', ' + pincite
					add_record(previous, sentence, path)
					break
		if is_id(sentence) and contains_usable_quote(previous):
			if i >= 2 and is_plain_cite_string(sentences[i-2]):
				add_record(previous, sentences[i-2], path)
			#print(join(previous, sentence))

def title_from_full_citation(s):
	match = re.match(FULL_CITATION_REGEX, s)
	return match.group(1)

def reporter_from_full_citation(s):
	match = re.match(FULL_CITATION_REGEX, s)
	parts = match.group(2).split()
	return ' '.join([parts[0], parts[1]])

def reporter_from_citation_without_name(s):
	match = re.match(CITATION_WITHOUT_NAME_REGEX, s)
	parts = match.group(1).split()
	return ' '.join([parts[0], parts[1]])

def reporter_title_pincite_from_short_cite(s):
	match = re.match(SHORT_CITATION_REGEX, s)
	return match.group(2), match.group(1), match.group(3)

def contains_usable_quote(s):
	if s.count('"') != 2:
		return False
	
	i = s.find('"')
	quote = s[i+1:]
	i = quote.find('"')
	quote = quote[:i]

	if '...' in quote or '. . .' in quote:
		return False
	
	if '[' in quote[1:]:
		return False
	
	return True

def add_record(sentence, citation, csv_path):
	if (match := re.match(FULL_CITATION_REGEX, citation)):
		case_title = match.group(1)
		case_cite = match.group(2)
	elif (match := re.match(CITATION_WITHOUT_NAME_REGEX, citation)):
		case_title = ''
		case_cite = match.group(1)

	field_names = ["sentence", "citation_sentence", "title", "citation", "quote"]
	i = sentence.find('"')
	quote = sentence[i+1:]
	i = quote.find('"')
	quote = quote[:i]
	quote = adjust_initial_capital(quote)
	record = {"sentence": sentence, "citation_sentence": citation,
		   "title": case_title, "citation": case_cite, "quote": quote}

	add_to_csv(csv_path, field_names, record)
	return record

def adjust_initial_capital(quote):
	if quote[0] == '[':
		if quote[1].islower():
			return quote[1].upper() + quote[3:]
		else:
			return quote[1].lower() + quote[3:]
	return quote

def has_one_quote(s):
	return s.count('"') == 1

def sentences_merge(sentences, i):
	new_sentences = sentences[:i]
	new_sentences.append(join(sentences[i], sentences[i+1]))
	new_sentences.extend(sentences[i+2:])
	return new_sentences

def join(a, b):
	return ' '.join([a, b])

def ends_with_t6(s):
	for e in BLUEBOOK_T6_PERIOD_ENDINGS:
		if s.endswith(e):
			return True
	return False

def is_plain_cite_string(s):
	if s.startswith("See"):
		return False
	match = re.match(FULL_CITATION_REGEX, s)
	return True if match else False

def is_cite_string_without_name(s):
	match = re.match(CITATION_WITHOUT_NAME_REGEX, s)
	return True if match else False

def is_short_cite(s):
	match = re.match(SHORT_CITATION_REGEX, s)
	return True if match else False

def is_id(s):
	match = re.match(r'^Id\.', s)
	return True if match else False

def clean_text(text):
	# Remove page numbers
	while (match := re.match(r'\n\d+ ?\n', text)):
		text = text[:match.start()] + text[match.end():]

	while (match := re.match(r'\*\d+', text)):
		text = text[:match.start()] + text[match.end():]

	# Remove excess whitespice
	text = text.replace('\n', ' ')
	for i in range(20):
		text = text.replace('  ', ' ')

	i = 0
	while (i := text.find('- ', i + 1)) > 0:
		if text[i-1].isalpha():
			text = text[:i] + text[i+2:]
	return text