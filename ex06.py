import pandas as pd
import seaborn as sns
import csv
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Category20c
from bokeh.transform import cumsum
from bokeh.layouts import column
from math import pi

print("Ran by Shashank Khanna\n")
print("-----PART 1-----")
# setting the maximum rows and columns for display.
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 22000)

# Reading the csv file
csv = pd.read_csv('SSE_Faculty.csv')
df = pd.DataFrame(csv)
df = df.fillna(0)    #replacing the Nan values in the dataframe to zero
#setting the ID to a list
data_id = df['ID'].values.tolist()

#adding into the variable the number of courses
a1 = (df[df['Program'] == 'EM']['Load 19-20'].sum())
print("\n\nNumber of courses in EM in the year 19-20 are : ",a1)

a2 = (df[df['Program'] == 'EM']['Load 20-21'].sum())
print("Number of courses in EM in the year 20-21 are : ",a2)

a3 = (df[df['Program'] == 'EM']['Load 21-22'].sum())
print("Number of courses in EM in the year 20-21 are : ",a3)

a4 = (df[df['Program'] == 'EM']['Load 22-23'].sum())
print("Number of courses in EM in the year 20-21 are : ",a4)

b1 = (df[df['Program'] == 'SSW']['Load 19-20'].sum())
print("\nNumber of courses in SSW in the year 19-20 are : ",b1)

b2 = (df[df['Program'] == 'SSW']['Load 20-21'].sum())
print("Number of courses in SSW in the year 20-21 are : ",b2)

b3 = (df[df['Program'] == 'SSW']['Load 21-22'].sum())
print("Number of courses in SSW in the year 21-22 are : ",b3)

b4 = (df[df['Program'] == 'SSW']['Load 22-23'].sum())
print("Number of courses in SSW in the year 22-23 are : ",b4)

c1 = (df[df['Program'] == 'SYS']['Load 19-20'].sum())
print("\nNumber of courses in SYS in the year 19-20 are : ",c1)

c2 = (df[df['Program'] == 'SYS']['Load 20-21'].sum())
print("Number of courses in SYS in the year 20-21 are : ",c2)

c3 = (df[df['Program'] == 'SYS']['Load 21-22'].sum())
print("Number of courses in SYS in the year 21-22 are : ",c3)

c4 = (df[df['Program'] == 'SYS']['Load 22-23'].sum())
print("Number of courses in SYS in the year 22-23 are : ",c4,"\n\n")

# calculating average number of courses assigned to each faculty for every academic year.
avg = list(df.iloc[:, [2, 5, 8, 11]].mean(axis=1))  # avg no of courses of each professor over the years
res = "\n".join("ID {} ->  {}".format(x, y) for x, y in zip(data_id, avg))
# Printing average number of courses per faculty
print("\nThe average number of courses for each faculty per year are: ")
average = df.iloc[:, [0,2, 5, 8, 11]]
print(average)

#using count() to find the number of ID having the condition as given below
e1 = (df[df['Balance 19-20'] < 0]['ID'].count())
print("\n\nNumber of underloaded faculty in the year 19-20 are : ",e1)

e2 = (df[df['Balance 20-21'] < 0]['ID'].count())
print("Number of underloaded faculty in the year 20-21 are : ",e2)

e3 = (df[df['Balance 21-22'] < 0]['ID'].count())
print("Number of underloaded faculty in the year 21-22 are : ",e3)

e4 = (df[df['Balance 22-23'] < 0]['ID'].count())
print("Number of underloaded faculty in the year 22-23 are : ",e4)

f1 = (df[df['Balance 19-20'] > 0]['ID'].count())
print("\n\nNumber of overloaded faculty in the year 19-20 are : ",f1)

f2 = (df[df['Balance 20-21'] > 0]['ID'].count())
print("Number of overloaded faculty in the year 20-21 are : ",f2)

f3 = (df[df['Balance 21-22'] > 0]['ID'].count())
print("Number of overloaded faculty in the year 21-22 are : ",f3)

f4 = (df[df['Balance 22-23'] > 0]['ID'].count())
print("Number of overloaded faculty in the year 22-23 are : ",f4)

print("\n\n-----PART 2-----")
print("\n Bokeh Plots")

output_file("output.html")

#LINE PLOT FOR Courses per program per Academic Year

A = int(a1 + a2 + a3 + a4)
B = int(b1 + b2 + b3 + b4)
C = int(c1 + c2 + c3 + c4)

# preparing the data
x = [ 1,2 ,3 , 4]
y1 = [a1,a2,a3,a4]
y2 = [b1,b2,b3,b4]
y3 = [c1,c2,c3,c4]
# create a new plot with a title and axis labels
p2 = figure(title="Courses per program per Academic Year", x_axis_label="Years", y_axis_label="Courses")

# add a line renderer with legend and line thickness
p2.line(x, y1, legend_label="EM", color="blue", line_width=2)
p2.line(x, y2, legend_label="SSW", color="red", line_width=2)
p2.line(x, y3, legend_label="SYS", color="green", line_width=2)

#BAR PLOT for Average number of courses per faculty over the years

# instantiating the figure object
graph = figure(title="Average number of courses per faculty over the years")

# name of the x-axis
graph.xaxis.axis_label = "ID"

# name of the y-axis
graph.yaxis.axis_label = "Number of Courses"

# width / thickness of the bars
width = 0.4

# plotting the graph ; data_id is the list with all the ID values & avg is the value of all average number of courses
graph.vbar(x = data_id,
           top=avg,
           width=width)

#LINE PLOT for Number of underloaded faculty over the years

x = [1 ,2 ,3 , 4]
y1 = [e1,e2,e3,e4]

# create a new plot with a title and axis labels
p1 = figure(title="Number of underloaded faculty over the years", x_axis_label="Years", y_axis_label="ID")

# add a line renderer with legend and line thickness
p1.line(x, y1, legend_label="Faculty", color="blue", line_width=2)

#PIE CHART PLOT for Courses by program in '22-'23

# counting the number of courses by programs in 22-23
em = df[df['Program'] == 'EM']['Load 22-23'].count()
sys = df[df['Program'] == 'SYS']['Load 22-23'].count()
ssw = df[df['Program'] == 'SSW']['Load 22-23'].count()

x = {"EM": em, "SYS": sys, "SSW": ssw}

data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'country'})
data['angle'] = data['value']/data['value'].sum() * 2*pi
data['color'] = Category20c[len(x)]

p = figure(height=500, title="Pie Chart", toolbar_location=None,
           tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

p.wedge(x=0, y=1, radius=0.5,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='country', source=data)

p.axis.axis_label = None
p.axis.visible = False
p.grid.grid_line_color = None

# show the results in BOKEH PLOTS
show(column(p2, graph, p1 , p))