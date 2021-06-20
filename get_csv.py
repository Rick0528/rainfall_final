import requests
from bs4 import BeautifulSoup
import csv

csvfile = "rainfall_year_data.csv" #開個csv檔案準備寫入

def get_column():
    url = "https://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain&station=467490&stname=%25E8%2587%25BA%25E4%25B8%25AD&datepicker=2021-05"
    r = requests.get(url)   
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text,"lxml")
    tag_table = soup.find(id="MyTable") #用BeautifulSoup找到table位置
    rows = tag_table.findAll("tr") #找到每個
    column = rows[1:3]

    with open(csvfile, "a+", newline="", encoding="utf-8-sig") as fp:       #utf-8-sig
        writer = csv.writer(fp)
        for row in column:
            rowList=[]
            for cell in row.findAll(["td","th"]):
                rowList.append(cell.get_text(strip=True).replace("\n","").replace("\r",""))
            writer.writerow(rowList)
        
def to_csv(url):
    r = requests.get(url)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "lxml")
    tag_table = soup.find(id="MyTable") #用BeautifulSoup找到table位置
    rows = tag_table.findAll("tr") #找到每個
    rows = rows[3:]

    with open(csvfile, "a+", newline="", encoding="utf-8-sig") as fp:       #utf-8-sig
        writer = csv.writer(fp)
        for row in rows:
            rowList=[]
            for cell in row.findAll(["td","th"]):
                text = cell.get_text(strip=True)
                rowList.append(text.replace("\n","").replace("\r",""))
            writer.writerow(rowList)

#月報表(每日)
# if __name__ == '__main__':
#     get_column()
#     for year in range(2000, 2021):
#         for month in range(1, 13):
#             if month < 10:
#                 month = str(0) + str(month)
#             url = f"https://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain&station=467490&stname=%25E8%2587%25BA%25E4%25B8%25AD&datepicker={year}-{month}"
#             print(url)
#             to_csv(url)

#年報表(每月)
if __name__ == '__main__':
    get_column()
    for year in range(2000, 2021):
        url = f"https://e-service.cwb.gov.tw/HistoryDataQuery/YearDataController.do?command=viewMain&station=467490&stname=%25E8%2587%25BA%25E4%25B8%25AD&datepicker={year}"
        print(url)
        to_csv(url)