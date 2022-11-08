import base64
import math
import sys
from io import BytesIO

import matplotlib
import uvicorn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import Request, APIRouter, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

router = APIRouter()
templates = Jinja2Templates(directory="templates/")


def group_by_min_max_mean(dataframe, min, max, rangemm, column, groupedByColumn):
    return dataframe.groupby(pd.cut(dataframe[column], np.arange(min, max + rangemm, rangemm)))[groupedByColumn].agg(['min', 'max', 'mean'])


@router.post("/parse", response_class=HTMLResponse)
def parse(
        request: Request,
        file_path: str = Form(...),
        row_start: int = Form(...), row_end: int = Form(...),
        column_start: int = Form(...), column_end: int = Form(...),
        amount_group: int = Form(...)
):
    try:
        matplotlib.use('TkAgg')
        dataframe = pd.read_csv("csv/" + file_path, sep=",")
        filtered_data = dataframe.iloc[row_start:row_end + 1, column_start:column_end + 1]
        dataframe2 = pd.DataFrame(filtered_data)

        min = dataframe2['Store_Area'].min()
        max = dataframe2['Store_Area'].max()
        range_store_area = (max - min) / amount_group

        # lab 2

        # 1
        items_available_grouped_by_store_area = \
            group_by_min_max_mean(dataframe2, min, max, range_store_area, "Store_Area", 'Items_Available')

        # 2
        daily_customer_count_grouped_by_store_area = \
            group_by_min_max_mean(dataframe2, min, max, range_store_area, "Store_Area", 'Daily_Customer_Count')

        # 3
        store_sales_grouped_by_store_area = \
            group_by_min_max_mean(dataframe2, min, max, range_store_area, "Store_Area", 'Store_Sales')

        # 4
        min = dataframe2['Daily_Customer_Count'].min()
        max = dataframe2['Daily_Customer_Count'].max()
        range_daily_customer_count = (max - min) / amount_group
        store_sales_grouped_by_daily_customer_count = \
            group_by_min_max_mean(dataframe2, min, max, range_daily_customer_count,
                                  "Daily_Customer_Count", 'Store_Sales')

        # lab 3
        mean_store_area = dataframe2['Store_Area'].mean()
        mean_items_available = dataframe2['Items_Available'].mean()
        mean_daily_customer_count = dataframe2['Daily_Customer_Count'].mean()
        mean_store_sales = dataframe2['Store_Sales'].mean()

        new_number_rows = round(len(dataframe2.index) * 0.1)

        for i in range(new_number_rows):
            store_id = len(dataframe2.index)
            new_row = {'Store ID ': int(store_id), 'Store_Area': mean_store_area,
                       'Items_Available': mean_items_available,
                       'Daily_Customer_Count': mean_daily_customer_count, 'Store_Sales': mean_store_sales}
            dataframe2 = dataframe2.append(new_row, ignore_index=True)

        filtered_data = dataframe2.iloc[row_start:row_end + 1, column_start:column_end + 1]

        # 1
        min = dataframe2['Store_Area'].min()
        max = dataframe2['Store_Area'].max()
        items_available_grouped_by_store_area2 = \
            group_by_min_max_mean(dataframe2, min, max, range_store_area, "Store_Area", 'Items_Available')

        # 2
        daily_customer_count_grouped_by_store_area2 = \
            group_by_min_max_mean(dataframe2, min, max, range_store_area, "Store_Area", 'Daily_Customer_Count')

        # 3
        store_sales_grouped_by_store_area2 = \
            group_by_min_max_mean(dataframe2, min, max, range_store_area, "Store_Area", 'Store_Sales')

        # 4
        min = dataframe2['Daily_Customer_Count'].min()
        max = dataframe2['Daily_Customer_Count'].max()
        range_daily_customer_count = (max - min) / amount_group
        store_sales_grouped_by_daily_customer_count2 = \
            group_by_min_max_mean(dataframe2, min, max, range_daily_customer_count,
                                  "Daily_Customer_Count", 'Store_Sales')

        items_available_grouped_by_store_area.plot.bar(rot=0)
        items_available_grouped_by_store_area2.plot.bar(rot=0)

        daily_customer_count_grouped_by_store_area.plot.bar(rot=0)
        daily_customer_count_grouped_by_store_area2.plot.bar(rot=0)

        store_sales_grouped_by_store_area.plot.bar(rot=0)
        store_sales_grouped_by_store_area2.plot.bar(rot=0)

        store_sales_grouped_by_daily_customer_count.plot.bar(rot=0)
        store_sales_grouped_by_daily_customer_count2.plot.bar(rot=0)

        # df4 = dataframe2[(dataframe2["Store_Area"] >= "{}".format(min) & dataframe2["Store_Area"] <= "{}".format(min + range_store_area))]
        min = dataframe2['Store_Area'].min()
        max = dataframe2['Store_Area'].max()

        range_store_area = (max - min) / amount_group
        df4 = dataframe2[(dataframe2["Store_Area"] <= min + range_store_area)]['Daily_Customer_Count']
        df5 = dataframe2[(dataframe2["Store_Area"] > min + range_store_area) & (
                    dataframe2["Store_Area"] <= min + range_store_area * 2)]['Daily_Customer_Count']
        df6 = dataframe2[(dataframe2["Store_Area"] > min + range_store_area * 2) & (
                    dataframe2["Store_Area"] <= min + range_store_area * 3)]['Daily_Customer_Count']

        fig = plt.figure()
        fig.set_size_inches(12, 10)
        data = [df4, df5, df6]
        plt.boxplot(data)

        fig_list = []
        for i in plt.get_fignums():
            tmpfile = BytesIO()
            plt.figure(i).savefig(tmpfile, format='png')
            encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            fig_list.append('<img src=\'data:image/png;base64,{}\'>'.format(encoded))

        result = [filtered_data.columns.values.tolist()]
        for row in filtered_data.values.tolist():
            result.append(row)

        dataframe.info(buf=open('tmp_pandas_df_info.txt', 'w'))  # save to txt
        contents = open("tmp_pandas_df_info.txt", "r")
        result_info = ""
        for lines in contents.readlines():
            result_info += ("<pre>" + lines + "</pre>\n")

        # laba 5

        dataframe = pd.read_csv("csv/" + file_path, sep=",")
        filtered_data = dataframe.iloc[row_start:row_end + 1, column_start:column_end + 1]
        dataframe3 = pd.DataFrame(filtered_data)
        x = dataframe3.loc[:, ['Store_Area']]
        y = dataframe3.loc[:, ['Store_Sales']]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)
        model = LinearRegression().fit(x_train, y_train)
        traing_set = round(model.score(x_train, y_train) * 100, 5)
        test_set = round(model.score(x_test, y_test) * 100, 5)

        plt.figure(figsize=(16, 9))
        plt.scatter(x_train, y_train, color='black')
        plt.scatter(x_test, y_test, color='red')
        plt.xlabel('Площадь магазина (фт^2)', fontweight='bold')
        plt.ylabel('Выручка ($)', fontweight='bold')
        plt.ylim(0, dataframe3.Store_Sales.max())
        plt.xlim(750, dataframe3.Store_Area.max() + 100)
        plt.plot(x_train, model.predict(x_train), color="green", linewidth=3)

        tmpfile = BytesIO()
        plt.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        plotx = '<img class="img" src=\'data:image/png;base64,{}\'>'.format(encoded)



    except Exception:
        return {"message": "There was an error uploading the file"}
    return templates.TemplateResponse('csv.html',
                                      context={
                                          'request': request,
                                          'filename': file_path,
                                          'row_start': row_start,
                                          'row_end': row_end,
                                          'col_start': column_start,
                                          'col_end': column_end,
                                          'result': result,
                                          'result_info': result_info,
                                          'min': min,
                                          'max': max,
                                          'range': range_daily_customer_count,
                                          'items_available_grouped_by_store_area': items_available_grouped_by_store_area.to_html(
                                              index_names=False, escape=False),
                                          'daily_customer_count_grouped_by_store_area': daily_customer_count_grouped_by_store_area.to_html(
                                              index_names=False, escape=False),
                                          'store_sales_grouped_by_store_area': store_sales_grouped_by_store_area.to_html(
                                              index_names=False, escape=False),
                                          'store_sales_grouped_by_daily_customer_count': store_sales_grouped_by_daily_customer_count.to_html(
                                              index_names=False, escape=False),
                                          'plot': fig_list,
                                          'traing_set': traing_set,
                                          'test_set': test_set,
                                          'plotx': plotx,
                                          'x_train': x_train,
                                          'x_test': x_test
                                      })
