def pingyu(df):
    "F1"
    f1 = sum([x for x in df])/len(df)
    "F2"
    f2  = math.pow(sum([(x-f1)**2 for x in df]),2)/(len(df)-1)
    "f3"
    f3 = sum([pow((x-f1),3) for x in df])/(pow(math.sqrt(f2),3)*len(df))
    "f4"
    f4 = sum([pow((x-f1),4) for x in df]) /pow(f2,2)*len(df)
    p_y = [round(f1,3),round(f2,3),round(f3,3),round(f4,3)]
    return p_y
if __name__ == '__main__':
    a=fft_y
    b = pingyu(a)
b
