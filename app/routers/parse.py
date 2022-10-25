import math
import sys

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import Request, APIRouter, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates/")


def add_range(filtered_data, rows, rangex, list, column):
    start = 0
    end = rangex - 1
    index = 0
    while end < rows:
        if end + rangex >= rows:
            while end < rows - 1:
                end += 1
        list[index].insert(0, f'{filtered_data.values[start][column]}-{filtered_data.values[end][column]}')
        start += rangex
        end += rangex
        index += 1
    return list


def grouped_list_by_column(filtered_data, rows, amount_groups, column, column_sorted):
    rangex = int(rows / amount_groups)
    list = [[0] * 3 for _ in range(amount_groups)]
    for group in list:
        group[0] = sys.maxsize
        group[1] = -1

    for rowX in range(rows):
        ind_group = int(rowX / rangex)
        if ind_group > len(list) - 1:
            ind_group = ind_group - 1
        group = list[ind_group]
        if group[0] > filtered_data.values[rowX][column]:
            group[0] = filtered_data.values[rowX][column]
        elif group[1] < filtered_data.values[rowX][column]:
            group[1] = filtered_data.values[rowX][column]
        group[2] += filtered_data.values[rowX][column]
    for group in range(len(list) - 1):
        list[group][2] /= rangex
    list[len(list) - 1][2] /= (rangex + len(list) % rangex - 1)
    list = add_range(filtered_data, rows, rangex, list, column_sorted)
    return list


@router.post("/parse", response_class=HTMLResponse)
def parse(
        request: Request, file: UploadFile = File(...),
        row_start: int = Form(...), row_end: int = Form(...),
        column_start: int = Form(...), column_end: int = Form(...),
        amount_group: int = Form(...)
):
    try:
        dataframe = pd.read_csv(file.file)
        filtered_data = dataframe.iloc[row_start:row_end + 1, column_start:column_end + 1]
        dataframe2 = pd.DataFrame(filtered_data)

        min = dataframe2['Store_Area'].min()
        max = dataframe2['Store_Area'].max()
        range_store_area = (max - min) / amount_group

        sorted_data = dataframe2.sort_values(by='Store_Area')

        # lab 2

        # 1
        items_available_grouped_by_store_area = \
            dataframe2.groupby(pd.cut(dataframe2["Store_Area"], np
                                      .arange(min, max + range_store_area,
                                              range_store_area)))['Items_Available'].agg(['min', 'max', 'mean'])

        # 2
        daily_customer_count_grouped_by_store_area = \
            dataframe2.groupby(pd.cut(dataframe2["Store_Area"], np
                                      .arange(min, max + range_store_area,
                                              range_store_area)))['Daily_Customer_Count'].agg(['min', 'max', 'mean'])

        # 3
        store_sales_grouped_by_store_area = \
            dataframe2.groupby(pd.cut(dataframe2["Store_Area"], np
                                      .arange(min, max + range_store_area,
                                              range_store_area)))['Store_Sales'].agg(['min', 'max', 'mean'])

        # 4
        min = dataframe2['Daily_Customer_Count'].min()
        max = dataframe2['Daily_Customer_Count'].max()
        range_daily_customer_count = (max - min) / amount_group
        store_sales_grouped_by_daily_customer_count = \
            dataframe2.groupby(pd.cut(dataframe2["Daily_Customer_Count"], np
                                      .arange(min, max + range_daily_customer_count,
                                              range_daily_customer_count)))['Store_Sales'].agg(['min', 'max', 'mean'])

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
            dataframe2.groupby(pd.cut(dataframe2["Store_Area"], np
                                      .arange(min, max + range_store_area,
                                              range_store_area)))['Items_Available'].agg(['min', 'max', 'mean'])

        # 2
        daily_customer_count_grouped_by_store_area2 = \
            dataframe2.groupby(pd.cut(dataframe2["Store_Area"], np
                                      .arange(min, max + range_store_area,
                                              range_store_area)))['Daily_Customer_Count'].agg(['min', 'max', 'mean'])

        # 3
        store_sales_grouped_by_store_area2 = \
            dataframe2.groupby(pd.cut(dataframe2["Store_Area"], np
                                      .arange(min, max + range_store_area,
                                              range_store_area)))['Store_Sales'].agg(['min', 'max', 'mean'])

        # 4
        min = dataframe2['Daily_Customer_Count'].min()
        max = dataframe2['Daily_Customer_Count'].max()
        range_daily_customer_count = (max - min) / amount_group
        store_sales_grouped_by_daily_customer_count2 = \
            dataframe2.groupby(pd.cut(dataframe2["Daily_Customer_Count"], np
                                      .arange(min, max + range_daily_customer_count,
                                              range_daily_customer_count)))['Store_Sales'].agg(['min', 'max', 'mean'])

        items_available_grouped_by_store_area.plot()
        items_available_grouped_by_store_area2.plot()

        daily_customer_count_grouped_by_store_area.plot()
        daily_customer_count_grouped_by_store_area2.plot()

        store_sales_grouped_by_store_area.plot()
        store_sales_grouped_by_store_area2.plot()

        store_sales_grouped_by_daily_customer_count.plot()
        store_sales_grouped_by_daily_customer_count2.plot()

        plt.show()

        """
        # lab 2 extra

        min = dataframe2['Store_Area'].min()
        max = dataframe2['Store_Area'].max()
        range_store_area = (max - min) / amount_group
        dataframe2['Result'] = dataframe2["Store_Sales"].div(dataframe2['Daily_Customer_Count'].values)
        store_sales_for_person_grouped_by_store_area = \
            dataframe2.groupby(pd.cut(dataframe2["Store_Area"], np
                                      .arange(min, max + range_store_area,
                                              range_store_area)))['Result'].agg(['min', 'max', 'mean'])

        dataframe3 = pd.DataFrame(daily_customer_count_grouped_by_store_area)
        dataframe4 = pd.DataFrame(store_sales_grouped_by_store_area)

        dataframe5 = pd.DataFrame()
        dataframe5['minD'] = dataframe4["min"].div(dataframe3['min'].values)
        dataframe5['maxD'] = dataframe4["max"].div(dataframe3['max'].values)
        dataframe5['meanD'] = dataframe4["mean"].div(dataframe3['mean'].values)
        """

        """
        # 1
        items_available_grouped_by_store_area_list = grouped_list_by_column(sorted_data, row_end + 1, amount_group, 2, 1)
        items_available_grouped_by_store_area_list.insert(0, ['Диапазон площади', 'Минимум', 'Максимум', 'Среднее'])

        # 2
        daily_customer_count_grouped_by_store_area_list \
            = grouped_list_by_column(sorted_data, row_end + 1, amount_group, 3, 1)
        daily_customer_count_grouped_by_store_area_list.insert(0, ['Диапазон площади', 'Минимум', 'Максимум', 'Среднее'])

        # 3
        store_sales_grouped_by_store_area_list = grouped_list_by_column(sorted_data, row_end + 1, amount_group, 4, 1)
        store_sales_grouped_by_store_area_list.insert(0, ['Диапазон площади', 'Минимум', 'Максимум', 'Среднее'])

        # 4
        sorted_data = dataframe2.sort_values(by='Daily_Customer_Count')
        store_sales_grouped_by_daily_customer_count_list \
            = grouped_list_by_column(sorted_data, row_end + 1, amount_group, 4, 3)
        store_sales_grouped_by_daily_customer_count_list.insert(0, ['Диапазон покупателей', 'Минимум', 'Максимум',
                                                                    'Среднее'])
        """

        # min = filtered_data.values[0][2]
        # min = store_sales_grouped_by_store_area_list

        result = [filtered_data.columns.values.tolist()]
        for row in filtered_data.values.tolist():
            result.append(row)

        dataframe.info(buf=open('tmp_pandas_df_info.txt', 'w'))  # save to txt
        contents = open("tmp_pandas_df_info.txt", "r")
        result_info = ""
        for lines in contents.readlines():
            result_info += ("<pre>" + lines + "</pre>\n")

        """
        'grouped_items': items_available_grouped_by_store_area_list,
        'grouped_customers': daily_customer_count_grouped_by_store_area_list,
        'grouped_sales_by_area': store_sales_grouped_by_store_area_list,
        'grouped_sales_by_customers': store_sales_grouped_by_daily_customer_count_list,
        
        
        'store_sales_for_person_grouped_by_store_area': store_sales_for_person_grouped_by_store_area.to_html(
                                              index_names=False, escape=False),
        'dataframe5': dataframe5.to_html(index_names=False, escape=False),
        """
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    plt.show()
    return templates.TemplateResponse('csv.html',
                                      context={
                                          'request': request,
                                          'filename': file.filename,
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
                                      })
