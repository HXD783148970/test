
def get_rms(records):
    
    """均方根值 反映的是有效值而不是平均值 """
    root_mean = math.sqrt(sum([x ** 2 for x in records]) / len(records))
    """峰峰值"""
    peak_to_peak = max(records)-min(records)
    """峰值指标"""
    crest_factor =  max(records)/root_mean
    """波形指标"""
    shape_factor = root_mean/abs(sum([x for x in records]) / len(records))
    """脉冲指标"""
    impulse_factor = max(records)/abs(sum([x for x in records]) / len(records))
    """裕度指标"""
    clarance =  max(records1)/pow(abs((sum(sqrt([abs(x) for x in records1]))/len(records1))),2)
    """峭度指标"""
    kur =  (sum([x**4 for x in records])/len(records))/pow(root_mean,4)
    pstf = [round(root_mean,3),round(peak_to_peak,3),round(crest_factor,3),round(shape_factor,3),round(impulse_factor,3),round(clarance,3),round(kur,3)]
    return pstf
if __name__ == '__main__':
    records1 = [1, 2, 3, 4, 5, 6]
    records2 = [2, 4, 6]
       # 均方根
    rms1 = get_rms(records1)  # 4.08
    rms2 = get_rms(records2)  # 4.32

rms1
