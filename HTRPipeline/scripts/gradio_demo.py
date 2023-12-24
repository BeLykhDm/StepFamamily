import json
import os
import threading
import time

import cv2
import gradio as gr
import pandas as pd

from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree

with open('../data/words_alpha.txt') as f:
    word_list = [w.strip().upper() for w in f.readlines()]
prefix_tree = PrefixTree(word_list)


def process_page(img, scale, margin, use_dictionary, min_words_per_line, text_scale):
    # read page
    read_lines = read_page(img,
                           detector_config=DetectorConfig(scale=scale, margin=margin),
                           line_clustering_config=LineClusteringConfig(min_words_per_line=min_words_per_line),
                           reader_config=ReaderConfig(decoder='word_beam_search' if use_dictionary else 'best_path',
                                                      prefix_tree=prefix_tree))

    # create text to show
    res = ''
    structured_data = []  # To hold structured data for DataFrame
    # tabel generator
    for read_line in read_lines:
        line_text = ' '.join(read_word.text for read_word in read_line)
        res += line_text + '\n'
        structured_data.append(line_text.split())  # Split line into words
    # create visualization to show
    for i, read_line in enumerate(read_lines):
        for read_word in read_line:
            aabb = read_word.aabb
            cv2.rectangle(img,
                          (aabb.xmin, aabb.ymin),
                          (aabb.xmin + aabb.width, aabb.ymin + aabb.height),
                          (255, 0, 0),
                          2)
            cv2.putText(img,
                        read_word.text,
                        (aabb.xmin, aabb.ymin + aabb.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_scale,
                        color=(255, 0, 0))
    with open('output.txt', 'w') as file:
        file.write(res)
    max_columns = max(len(row) for row in structured_data)

    # Standardize the length of each row
    standardized_data = [row + [''] * (max_columns - len(row)) for row in structured_data]

    # Create string headers for the DataFrame
    headers = [f'Column {i + 1}' for i in range(max_columns)]

    # Create the DataFrame with standardized rows and string headers
    df = pd.DataFrame(standardized_data, columns=headers)
    return res, img, df


with open('../data/config.json') as f:
    config = json.load(f)

examples = []
for k, v in config.items():
    examples.append([f'../data/{k}', v['scale'], v['margin'], False, 2, v['text_scale']])

# define gradio interface
interface = gr.Interface(fn=process_page,
             inputs=[gr.Image(label='Input image'),
                     gr.Slider(0, 10, 1, step=0.01, label='Scale'),
                     gr.Slider(0, 25, 1, step=1, label='Margin'),
                     gr.Checkbox(value=False, label='Use dictionary'),
                     gr.Slider(1, 10, 1, step=1, label='Minimum number of words per line'),
                     gr.Slider(0.5, 2, 1, label='Text size in visualization')],
             outputs=[gr.Textbox(label='Read Text'), gr.Image(label='Visualization'), gr.Dataframe(label='Extracted Data')],
             live = True,
             examples=examples,
             allow_flagging='never',
             title='Detect and Read Handwritten Words',
             theme=gr.themes.Monochrome())

def monitor_output():
    last_size = 0
    while True:
        try:
            current_size = os.path.getsize('output.txt')
            if current_size != last_size:
                with open('output.txt', 'r') as file:
                    data = file.readlines()
                # Assuming each line is a row and words are columns
                structured_data = [line.split() for line in data]

                # Create a DataFrame
                df = pd.DataFrame(structured_data)
                print(df)
                last_size = current_size
        except FileNotFoundError:
            pass

        time.sleep(1)


# Run the monitoring function in a separate thread
monitor_thread = threading.Thread(target=monitor_output, daemon=True)
monitor_thread.start()
output_data = interface.launch(share = True)






