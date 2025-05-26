from graphviz import Digraph

# 创建一个图对象，设置输出格式为png
dot = Digraph(comment='E - R Diagram', format='png')
dot.engine = 'dot'  # 设置布局引擎

# 添加实体“缴纳罚金管理”，用矩形表示
dot.node('缴纳罚金管理', shape='rect')

# 定义属性列表
attributes = ['罚款说明', '罚款金额', '罚款日期', '图书名称', '图书编号', '借阅单号', '用户名', '手机', '是否支付']

# 循环添加属性节点（椭圆形）并连接到实体
for attr in attributes:
    dot.node(attr, shape='ellipse')
    dot.edge('缴纳罚金管理', attr)

# 渲染并查看E - R图
dot.render('fine_payment_er', view=True)