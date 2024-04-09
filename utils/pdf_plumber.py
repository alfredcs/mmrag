from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure
import pdfplumber


import requests
import pandas as pd
from tqdm import tqdm


import re


def process_page(page, extracted_page, text=True, table=True):
   content = []
  
   # Find the tables in the page
   tables = page.find_tables()
   extracted_tables = page.extract_tables()


   table_num = 0
   first_table_element = True
   table_extraction_process = False


   # Get a sorted list of elements based on their Y-coordinate in reverse order
   elements = [element for element in extracted_page._objs]
   elements.sort(key=lambda a: a.y1, reverse=True)


   lower_side = 0
   upper_side = 0
   for i, element in enumerate(elements):
       # Extract text if the element is a text container and text extraction is enabled
       if isinstance(element, LTTextContainer) and not table_extraction_process and text:
           line_text = text_extraction(element)
           content.append(line_text)


       # Process tables if the element is a rectangle and table extraction is enabled
       if isinstance(element, LTRect) and table:
           if first_table_element and table_num < len(tables):
               lower_side = page.bbox[3] - tables[table_num].bbox[3]
               upper_side = element.y1


               table = extracted_tables[table_num]
               table_string = convert_table(table)
               content.append(table_string)
               table_extraction_process = True
               first_table_element = False


           # Check if we have already extracted the tables from the page
           if element.y0 >= lower_side and element.y1 <= upper_side:
               pass
           elif i + 1 >= len(elements):
               pass
           elif not isinstance(elements[i + 1], LTRect):
               table_extraction_process = False
               first_table_element = True
               table_num += 1


   # Combine and clean up the extracted content
   content = re.sub('\n+', '\n', ''.join(content))
   return content

def normalize_text(line_texts):
   norm_text = ''
   for line_text in line_texts:
       line_text=line_text.strip()
       # empty strings after striping convert to newline character
       if not line_text:
           line_text = '\n'
       else:
           line_text = re.sub('\s+', ' ', line_text)
           # if the last character is not a letter or number, add newline character to a line
           if not re.search('[\w\d\,\-]', line_text[-1]):
               line_text+='\n'
           else:
               line_text+=' '
       # concatenate into single string
       norm_text+=line_text
   return norm_text


def text_extraction(element):
   # Extract text from line and split it with new lines
   line_texts = element.get_text().split('\n')
   line_text = normalize_text(line_texts)
   return line_text


def convert_table(table):
   table_string = ''
   # iterate through rows in the table
   for row in table:
       # clean row from newline character
       cleaned_row = [
           'None' if item is None else item.replace('\n', ' ')
           for item in row
       ]
       # concatenate the row as a string with the whole table
       table_string += f"|{'|'.join(cleaned_row)}|\n"
   return table_string.rstrip('\n')

def process_document(pdf_path, text=True, table=True, page_ids=None):
   pdf = pdfplumber.open(pdf_path)
   pages = pdf.pages
  
   # Extract pages from the PDF
   extracted_pages = extract_pages(pdf_path, page_numbers=page_ids)
  
   page2content = {}
  
   # Process each extracted page
   for extracted_page in tqdm(extracted_pages):
       page_id = extracted_page.pageid
       content = process_page(pages[page_id - 1], extracted_page, text, table)
       page2content[page_id] = content
  
   return page2content

if __name__ == "__main__":
    pdf_path = '/tmp/35-2-35.pdf'
    page2content = process_document(pdf_path, page_ids=[])
    print(page2content)