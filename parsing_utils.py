import re


def roman_to_arabic(roman):
    if roman is None: return None

    roman_numerals = {
        'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10,
        'XL': 40, 'L': 50, 'XC': 90, 'C': 100,
        'CD': 400, 'D': 500, 'CM': 900, 'M': 1000
    }

    # Convert to uppercase to handle lowercase input
    roman = roman.replace('(','').replace(')','').upper()
    i = 0
    num = 0

    while i < len(roman):
        # If the current and next characters form a valid numeral
        if i + 1 < len(roman) and roman[i:i+2] in roman_numerals:
            num += roman_numerals[roman[i:i+2]]
            i += 2
        else:
            num += roman_numerals[roman[i]]
            i += 1

    return int(num)
def parse_file(file_path):
    # Initialize the list to hold parsed data
    parsed_data = []

    # Read the file content
    with open(file_path, 'r',encoding="utf-8") as file:
        lines = file.readlines()

    # Regular expression to match section headers like "(i) AUTHOR"
    section_header_pattern = re.compile(r'^(\([^)]+\)) ([A-Z\s]+)(?: (\([^)]+\)))?$')

    current_header = None
    current_excerpt = []
    excerpt_id = None
    current_author = None
    rel = None

    for line in lines:
        line = line.strip()
        #print(line)
        # Check if the line matches the section header pattern
        match = section_header_pattern.match(line)
        if match:
            parsed_data.append({
                    'excerpt_id': roman_to_arabic(excerpt_id),
                    'author': current_author,
                    'excerpt': ' '.join(current_excerpt),
                    'relevant': roman_to_arabic(rel),
                    'doc_size': len(' '.join(current_excerpt))
                })
            current_excerpt = []
            # Update the current author
            current_header = match.group(0)
            excerpt_id =  match.group(1)
            current_author = match.group(2)
            if match.group(3):
                rel = match.group(3)
            else:
                rel = None
                #print("Rel not exits")
        elif line:
            # Add non-empty lines to the current excerpt
            #line =  re.sub(r'\’|\…|\.+|\!|\;|\:|\[|\]|\s{2,}|\d+', "", line).lower()
            line =  re.sub(r'\…|\.+|\!|\;|\:|\[|\]|\s{2,}|\d+', "", line).lower()

            current_excerpt.append(line)

    # Save the last author's data if applicable
    if current_header:
        parsed_data.append({
            'excerpt_id': roman_to_arabic(excerpt_id),
            'author': current_author,
            'excerpt': ' '.join(current_excerpt),
            'relevant': roman_to_arabic(rel),
            'doc_size': len(' '.join(current_excerpt))
        })

    return parsed_data


