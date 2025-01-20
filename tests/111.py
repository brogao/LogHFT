
from datetime import datetime
def convert_date(date):
    date_ob = datetime.strptime(date, '%Y-%m-%d')
    formatted_date = date_ob.strftime('%d/%m/%Y')
    return formatted_date


date=input()
converted_date = convert_date(date)
print(converted_date)

