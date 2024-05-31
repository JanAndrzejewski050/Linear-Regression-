import csv
import random

# growing tendency 
def generate_data(rows_num, data_labels, deviation_y, deviation_x, start_y, start_x,):
    data = [data_labels]      # m_w - X, m_s - Y
    x = start_x
    y = start_y
    for i in range (rows_num):
        r_x = random.randint(-int(deviation_x*0.3), int(deviation_x*0.7))
        r_y = random.randint(-int(deviation_y*0.35), int(deviation_y*0.65))
        x += r_x; x = abs(x)
        y += r_y; y = abs(y)
        data.append([x, y])
    return data
    
data = generate_data(25, ["mouse_size", "mouse_weight"], 8, 5, 15 , 10)

with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
