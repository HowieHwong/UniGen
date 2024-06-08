import json
from transformers import pipeline


class NERProcessor:
    def __init__(self):
        """Initialize the NER model."""
        self.ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

    def extract_entities(self, sentence):
        """Extract named entities from a sentence using a pre-trained BERT model."""
        entities = self.ner_model(sentence)
        return entities

    @staticmethod
    def simplify_ner_results(ner_results):
        """Simplify the NER results by combining subwords and organizing them by entity type."""
        entities = {}
        current_entity = None

        for item in ner_results:
            entity_type = item['entity'][2:]  # Remove the I- or B- prefix
            word = item['word'].replace("##", "")  # Remove BERT's subword marker

            if current_entity is None or current_entity['type'] != entity_type:
                # Start a new entity if there's no current entity or the type has changed
                if current_entity is not None:
                    # Save the previous entity
                    entity_name = " ".join(current_entity['word'])
                    if current_entity['type'] not in entities:
                        entities[current_entity['type']] = [entity_name]
                    else:
                        entities[current_entity['type']].append(entity_name)

                current_entity = {
                    'type': entity_type,
                    'word': [word]
                }
            else:
                current_entity['word'].append(word)

        # Don't forget to add the last entity
        if current_entity is not None:
            entity_name = " ".join(current_entity['word'])
            if current_entity['type'] not in entities:
                entities[current_entity['type']] = [entity_name]
            else:
                entities[current_entity['type']].append(entity_name)

        return entities

    def process_json_file(self, input_filepath, output_filepath):
        """Process a JSON file to extract and simplify named entities for each sentence."""
        with open(input_filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for item in data:
            sentence = item['text']
            ner_results = self.extract_entities(sentence)
            print("NER results:")
            print(ner_results)
            simplified_entities = self.simplify_ner_results(ner_results)
            print("Simplified entities:")
            print(simplified_entities)
            item['entity list'] = simplified_entities

        with open(output_filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)


def load_entities(json_file_path):
    """ Load entities from a given JSON file path. """
    entities = set()
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        for item in data:
            entity_list = item.get("entity list", {})
            for entity_group in entity_list.values():
                entities.update(entity_group)
    return entities



def calculate_overlapped_ratio(file1, file2):
    # Load entities from both files
    entities_file1 = load_entities(file1)
    entities_file2 = load_entities(file2)

    # Calculate the intersection of entities
    intersection = entities_file1.intersection(entities_file2)

    # Calculate the overlap ratio
    overlap_ratio = len(intersection) / min(len(entities_file1), len(entities_file2)) if min(len(entities_file1), len(entities_file2)) > 0 else 0
    print("Overlap ratio:", overlap_ratio)
    return overlap_ratio



if __name__ == '__main__':
    processor = NERProcessor()
    file_list = ['original_file.json', 'new_file.json']
    processor.process_json_file('original_file.json', 'original_output.json')
    processor.process_json_file('new_file.json', 'new_output.json')
    calculate_overlapped_ratio('original_output.json', 'new_output.json')

