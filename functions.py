import pandas as pd
def create_dataframe_from_boxes(boxes):
    data = []
    for box in boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.data[0].tolist()[0:4]
        data.append([cls, conf, x1, y1, x2, y2])
    df = pd.DataFrame(data, columns=['Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
    return df