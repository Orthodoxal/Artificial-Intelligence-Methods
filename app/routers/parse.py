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


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=20, min_samples=10):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = []

    def fit(self, X, y):
        self.tree = self.grow_tree(X, y)

    def predict(self, X):
        return np.array([self.travers_tree(x, self.tree) for x in X])

    def most_common(self, y):
        return np.sum(y) / len(y)

    def entropy(self, y):
        predict = np.sum(y) / len(y)
        mse = np.sum((predict - y) ** 2) / len(y)
        mae = np.sum(np.abs(predict - y)) / len(y)
        return mae

    def best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_gain = -1

        for i in range(X.shape[1]):
            thresholds = np.unique(X[:, i])
            for threshold in thresholds:
                gain = self.information_gain(X[:, i], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_threshold = threshold
        return best_feature, best_threshold

    def information_gain(self, X_column, y, threshold):
        n = len(y)
        parent = self.entropy(y)

        left_indexes = np.argwhere(X_column <= threshold).flatten()
        right_indexes = np.argwhere(X_column > threshold).flatten()

        child = 0

        if len(left_indexes) != 0:
            e_l, n_l = self.entropy(y[left_indexes]), len(left_indexes)
            child += (n_l / n) * e_l
        if len(right_indexes) != 0:
            e_r, n_r = self.entropy(y[right_indexes]), len(right_indexes)
            child += (n_r / n) * e_r

        return parent - child

    def grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]

        if n_samples <= self.min_samples or depth >= self.max_depth:
            return Node(value=self.most_common(y))

        best_feature, best_threshold = self.best_split(X, y)

        left_indexes = np.argwhere(X[:, best_feature] <= best_threshold).flatten()
        right_indexes = np.argwhere(X[:, best_feature] > best_threshold).flatten()

        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return Node(value=self.most_common(y))

        left = self.grow_tree(X[left_indexes, :], y[left_indexes], depth + 1)
        right = self.grow_tree(X[right_indexes, :], y[right_indexes], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def travers_tree(self, x, tree):
        if tree.is_leaf_node():
            return tree.value

        if x[tree.feature] <= tree.threshold:
            return self.travers_tree(x, tree.left)
        return self.travers_tree(x, tree.right)


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

        x_train_99 = np.array(x_train)
        y_train_99 = np.array(y_train)
        size_data = np.size(x_train_99)
        # среднее арифметическое
        mean_x = np.mean(x_train_99)
        mean_y = np.mean(y_train_99)
        # сумма перекрестных отклонений y и x
        SS_xy = np.sum(y_train_99 * x_train_99) - size_data * mean_y * mean_x
        # сумма квадратов отклонений от x
        SS_xx = np.sum(x_train_99 * x_train_99) - size_data * mean_x * mean_x

        b = SS_xy / SS_xx
        a = mean_y - b * mean_x

        x_test_1 = np.array(x_test)
        y_test_1 = np.array(y_test)
        pred_y = a + b * x_test_1

        u = ((y_test_1 - pred_y) ** 2).sum()
        v = ((y_test_1 - y_test_1.mean()) ** 2).sum()
        R2 = 1 - u / v

        plt.figure(figsize=(16, 9))
        plt.scatter(x_train, y_train, color='black')
        plt.scatter(x_test, y_test, color='red')
        plt.xlabel('Площадь магазина (фт^2)', fontweight='bold')
        plt.ylabel('Выручка ($)', fontweight='bold')
        plt.ylim(0, dataframe3.Store_Sales.max())
        plt.xlim(750, dataframe3.Store_Area.max() + 100)
        y_pred = a + b * x_train
        plt.plot(x_train, y_pred, color="green", linewidth=3)
        # plt.plot(x_train, model.predict(x_train), color="green", linewidth=3)

        tmpfile = BytesIO()
        plt.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        plotx = '<img class="img" src=\'data:image/png;base64,{}\'>'.format(encoded)

        # laba 6

        x = dataframe3.loc[:, ['Store_Area', 'Daily_Customer_Count']]
        y = dataframe3.loc[:, ['Store_Sales']]

        x1_train, x1_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.1)
        clf = DecisionTree()
        clf.fit(x1_train, y_train)

        predicted = clf.predict(x1_test)
        print(predicted)
        print("////")
        print(y_test)
        print("////")
        u = ((predicted - y_test)**2).sum()
        v = ((y_test.mean() - y_test)**2).sum()
        R2 = 1 - u / v
        print(R2)




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
                                          'R2': R2,
                                          'plotx': plotx,
                                          'x_train': x_train,
                                          'x_test': x_test
                                      })
