import matplotlib.pyplot
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.ticker as mticker


def qp_size():
    X = [2*i for i in range(1,25)]
    Y = [16.3, 14.0, 11.2, 8.92, 6.69,
         5.33, 4.26, 3.54, 3.08, 2.66,
         2.29, 1.94, 1.62, 1.34, 1.08,
         0.914, 0.754, 0.632, 0.537, 0.469,
         0.414, 0.370, 0.34, 0.31]

    plt.plot(
        X, Y, '-p', color='grey',
        marker = 'o',
        markersize=4, linewidth=2,
        # markerfacecolor='blue',
        markeredgecolor='mediumpurple',
        markeredgewidth=2)
    plt.xlabel('QP', fontsize=14)
    plt.ylabel('Video size (MB)', fontsize=10)
    # plt.title(r'$f(x)=\sqrt{x},x,x^2,x^3$', fontsize=18)
    plt.show()


def resolution_F1():
    X = [0, 0.15, 0.84, 1, 0.68, 0.058, 0.12, 0.04, 0.095, 0.0064, 0.0041]
    Y = [1, 0, 0.85, 0.31, 0.57, 0.97, 0.97, 0.99, 0.98, 0.92, 0.99]
    dic = dict(zip(X, Y))
    print(dic)
    sort = sorted(dic.items(), key=lambda d: d[0], reverse=False)
    print(sort)
    x = []
    y = []
    for i in sort:
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x, y, s=50)
    plt.show()


def qp_F1():
    X = [2*i for i in range(1,25)]
    # precision = [1.0, 0.9938907980145093, 0.9908779931584949, 0.9931087289433385, 0.9777274443186108,
    #              0.9866666666666667, 0.9840304182509506, 0.9839755818389927, 0.9809451219512195, 0.977540921202893,
    #              0.9779551501330292, 0.9615528182157521, 0.9575825825825826, 0.9322897706589006, 0.944254835039818,
    #              0.9380934295480441, 0.9217877094972067, 0.9137737961926092, 0.9195884146341463, 0.9427312775330396,
    #              0.8977812995245642, 0.8980392156862745, 0.8382480485689505, 0.8393881453154876]
    #
    # recall = [1.0, 0.9908641035401599, 0.9923867529501332, 0.9874381423677199, 0.9859154929577465, 0.9859154929577465,
    #           0.9851541682527598, 0.9817282070803197, 0.9798248953178531, 0.977540921202893, 0.9794442329653598,
    #           0.9805862200228398, 0.9710696612105063, 0.9748762847354396, 0.9478492577084127, 0.9402360106585459,
    #           0.9421393224210126, 0.9318614389036924, 0.9185382565664256, 0.8960791777693187, 0.8625808907499048,
    #           0.7845451084887705, 0.7358203273696231, 0.6684430909783022]
    #
    # F1 = [1.0, 0.9923751429660693, 0.99163179916318, 0.9902653178087422, 0.9818043972706596, 0.9862909367859863,
    #       0.9845919726079514, 0.9828506097560975, 0.980384688630737, 0.977540921202893, 0.9786991251426397,
    #       0.9709762532981531, 0.9642789642789643, 0.9531075548939336, 0.946048632218845, 0.9391634980988594,
    #       0.9318524096385542, 0.9227289860535244, 0.9190630356122643, 0.9188134270101483, 0.879829159386527,
    #       0.8374644453474197, 0.783701601459558, 0.7442254714981987]

    precision = [1.0, 0.9939121228163049, 0.9896414342629483, 0.991462113127001, 0.982237539766702, 0.9844337090713903, 0.9841269841269841, 0.9738755723134931, 0.9767630370170224, 0.9685919616715465, 0.9716446124763705, 0.9569377990430622, 0.9461620469083155, 0.9344989561586639, 0.9424324324324325, 0.9223971278652306, 0.8911473656057769, 0.8740053050397878, 0.9250220523375478, 0.8969039259495691, 0.9220231822971549, 0.8995867768595042, 0.873639375295788, 0.7209098862642169]
    recall = [1.0, 0.990503824848325, 0.9828541281983646, 0.9802163017673438, 0.9773146926932208, 0.9675547348984437, 0.9649169084674228, 0.9538380374571354, 0.9535742548140332, 0.9599050382484833, 0.9490899498812978, 0.9496175151675019, 0.9364283830123978, 0.9446056449485624, 0.9198100764969664, 0.8810340279609602, 0.8789237668161435, 0.8691638090213664, 0.8298601951991559, 0.7412292271168557, 0.6924294381429702, 0.5742548140332366, 0.48694275916644686, 0.43471379583223424]
    F1 = [1.0, 0.9922050469018365, 0.9862361037586024, 0.9858071362249635, 0.979769932566442, 0.9759212451775975, 0.9744272775705913, 0.9637526652452025, 0.9650293646556327, 0.9642289348171701, 0.9602348545503069, 0.9532636038660134, 0.9412700517035664, 0.9395251213433031, 0.9309838472834068, 0.901241230437129, 0.8849933598937584, 0.8715778336198914, 0.874860956618465, 0.8116695551704218, 0.7909008737571558, 0.7010143294155531, 0.6253387533875339, 0.5423728813559322]

    plt.plot(X, precision, color="lightcoral", linewidth=2, linestyle="-", label="precision")
    plt.plot(X, recall, color="burlywood", linewidth=2, linestyle="--", label="recall")
    plt.plot(X, F1, color="mediumturquoise", linewidth=2, linestyle="-.", label="F1")
    plt.legend(loc="best")
    plt.xlabel('QP', fontsize=10)
    plt.ylabel('precenatge', fontsize=10)
    plt.show()


def QP_codeRate():
    X = [2*i for i in range(1, 25)]
    Y = [13583, 11661, 9326, 7348, 5478, 4337, 3438, 2835, 2448, 2095, 1789, 1497, 1223, 990, 775, 611, 480, 380, 302,
         247, 201, 166, 140, 116]
    plt.plot(
        X, Y, '-p', color='grey',
        marker='o',
        markersize=4, linewidth=2,
        # markerfacecolor='blue',
        markeredgecolor='mediumpurple',
        markeredgewidth=2)
    plt.xlabel('QP', fontsize=14)
    plt.ylabel('Video coding rate (kbps)', fontsize=10)
    plt.show()


def qp_codeRate_F1():
    X = [2 * i for i in range(1, 25)]
    F1 = [1.0, 0.9922050469018365, 0.9862361037586024, 0.9858071362249635, 0.979769932566442, 0.9759212451775975,
          0.9744272775705913, 0.9637526652452025, 0.9650293646556327, 0.9642289348171701, 0.9602348545503069,
          0.9532636038660134, 0.9412700517035664, 0.9395251213433031, 0.9309838472834068, 0.901241230437129,
          0.8849933598937584, 0.8715778336198914, 0.874860956618465, 0.8116695551704218, 0.7909008737571558,
          0.7010143294155531, 0.6253387533875339, 0.5423728813559322]
    codeRate = [13583, 11661, 9326, 7348, 5478, 4337, 3438, 2835, 2448, 2095, 1789, 1497, 1223, 990, 775, 611, 480, 380, 302,
         247, 201, 166, 140, 116]
    for index, item in enumerate(codeRate):
        codeRate[index] = item/1024
    font = {'family': 'serif',
            'serif': 'Times New Roman',
            'weight': 'normal',
            'size': 10}
    plt.rc('font', **font)
    fig, ax1 = plt.subplots(figsize=(8, 6))
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }

    ax1.tick_params(labelsize=18)
    plt.plot(X, F1, 'mediumturquoise', label="Accuracy(F1 score)", linewidth=2)
    # 显示网格
    plt.grid(True)
    plt.xlabel("QP", font1)
    plt.ylabel('Accuracy(F1 score)', font1)
    # plt.title("This is double axis label")
    # 设置线标的位置
    plt.legend(loc='lower left', fontsize=18)

    # 第二纵轴的设置和绘图
    ax2 = ax1.twinx()
    # ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d Mbps'))
    plt.plot(X, codeRate, 'burlywood', label="Bitrate", linewidth=2)
    plt.legend(loc='upper right', fontsize=18)
    ax2.tick_params(labelsize=18)
    ax2.set_ylabel("Bitrate (Mbps)", font1)
    # 限制横轴显示刻度的范围
    # plt.xlim(0, 24)
    plt.show()


def codeRate_F1():
    X = [13583, 11661, 9326, 7348, 5478, 4337, 3438, 2835, 2448, 2095, 1789, 1497, 1223, 990, 775, 611, 480, 380, 302,
         247, 201, 166, 140, 116]
    max = X[0]
    for index in range(len(X)):
        X[index] = X[index]/max
    Y = [1.0, 0.9923751429660693, 0.99163179916318, 0.9902653178087422, 0.9818043972706596, 0.9862909367859863,
         0.9845919726079514, 0.9828506097560975, 0.980384688630737, 0.977540921202893, 0.9786991251426397,
         0.9709762532981531, 0.9642789642789643, 0.9531075548939336, 0.946048632218845, 0.9391634980988594,
         0.9318524096385542, 0.9227289860535244, 0.9190630356122643, 0.9188134270101483, 0.879829159386527,
         0.8374644453474197, 0.783701601459558, 0.7442254714981987]
    plt.plot(
        X, Y, '-p', color='grey',
        marker='o',
        markersize=4, linewidth=2,
        # markerfacecolor='blue',
        markeredgecolor='mediumpurple',
        markeredgewidth=2)
    plt.xlabel('Video coding rate (kbps)', fontsize=10)
    plt.ylabel('F1', fontsize=14)
    plt.show()
    index = 1
    a = 1
    min = a*X[index]-Y[index]
    for i in range(2, len(X)):
        temp = a*X[i]-Y[i]
        if temp < min:
            min = temp
            index = i
    print(min)
    print(index)


def partQP_codingRate():
    X = [2 * i for i in range(9, 25)]
    Y = [655, 651, 640, 637, 631, 626, 619, 610, 605, 600, 608, 592, 599, 589, 579, 570]
    plt.plot(
        X, Y, '-p', color='grey',
        marker='o',
        markersize=4, linewidth=2,
        # markerfacecolor='blue',
        markeredgecolor='mediumpurple',
        markeredgewidth=2)
    plt.xlabel('part QP', fontsize=14)
    plt.ylabel('Video coding rate (kbps)', fontsize=10)
    plt.show()


def partQP_F1():
    X = [2 * i for i in range(1, 25)]
    precision = [0.9140560022179096, 0.9078265612217071, 0.9154228855721394, 0.9070024910047052, 0.9121964991530209,
     0.9103064066852368, 0.913570052298376, 0.9172763083849184, 0.9066965536564864, 0.9072079536039768,
     0.9126625211984172, 0.8875968992248062, 0.8974936637566883, 0.8873279816513762, 0.8802127064091799,
     0.8974056603773585, 0.8902728351126927, 0.8942368587713743, 0.8694054776219104, 0.8139386189258312,
     0.8458301453710788, 0.7970878346641616, 0.7684463107378524, 0.6855600539811066]
    recall = [0.8696913743075706, 0.8781324188868372, 0.8736481139541018, 0.8644157214455289, 0.8522817198628331,
     0.8620416776576101, 0.8754945924558164, 0.8599314165127935, 0.8536006330783434, 0.8665259825903455,
     0.8517541545766288, 0.8456871537852809, 0.8406752835663414, 0.8164072804009496, 0.8295964125560538,
     0.8029543656027434, 0.7918754945924558, 0.7449221841202849, 0.6866262199947244, 0.6715906093379056,
     0.5832234238987075, 0.4476391453442363, 0.33790556581376946, 0.2680031653917172]
    F1 = [0.8913219789132198, 0.8927326360954678, 0.8940477797273586, 0.8851971907077256, 0.8812218737215328,
     0.8855168676331119, 0.8941271551724137, 0.8876786929884275, 0.8793478260869565, 0.886400431732326,
     0.8811570473461591, 0.8661353505335674, 0.8681558158539907, 0.8503915372990797, 0.8541553503530689,
     0.84755673117082, 0.8381962864721485, 0.812778817095985, 0.7672807663964628, 0.735944500650383, 0.6903981264637002,
     0.5733108108108107, 0.4694027116159766, 0.3853593779632088]

    plt.plot(X, precision, color="lightcoral", linewidth=2, linestyle="-", label="precision")
    plt.plot(X, recall, color="burlywood", linewidth=2, linestyle="--", label="recall")
    plt.plot(X, F1, color="mediumturquoise", linewidth=2, linestyle="-.", label="F1")
    plt.legend(loc="best")
    plt.xlabel('part QP', fontsize=10)
    plt.ylabel('precenatge', fontsize=10)
    plt.show()


def mostQP_codingRate():
    X = [2*i for i in range(9, 25)]
    Y = [1638, 1624, 1537, 1458, 1358, 1244, 1110, 1020, 925, 841, 777,
         749, 704, 681, 658, 650]
    plt.plot(
        X, Y, '-p', color='grey',
        marker='o',
        markersize=4, linewidth=2,
        # markerfacecolor='blue',
        markeredgecolor='mediumpurple',
        markeredgewidth=2)
    plt.xlabel('most QP', fontsize=14)
    plt.ylabel('Video coding rate (kbps)', fontsize=10)
    plt.show()


def mostQP_F1():
    X = [2 * i for i in range(1, 25)]
    precision = [0.9576872536136662, 0.9622282608695653, 0.9560819803034336, 0.9546681057744199, 0.9656468062265163,
     0.9584895554365291, 0.9695006747638326, 0.9695006747638326, 0.9595687331536388, 0.9659434221367756,
     0.9465608465608466, 0.9543869063590019, 0.9455221897422269, 0.9426360725720384, 0.9661436829066887,
     0.9233766233766234, 0.9284775177491454, 0.9222689075630253, 0.9550898203592815, 0.9418381344307271,
     0.9488651900464862, 0.9637410071942446, 0.9695282742455318, 0.9140560022179096]

    recall = [0.9612239514639936, 0.934054339224479, 0.9475072540226853, 0.9332629912951728, 0.9490899498812978,
     0.9440780796623582, 0.9475072540226853, 0.9475072540226853, 0.9390662094434187, 0.927723555790029,
     0.9438142970192561, 0.9382748615141123, 0.9385386441572144, 0.9319440780796624, 0.9258770772883145,
     0.9377472962279082, 0.9314165127934582, 0.9264046425745186, 0.9256132946452124, 0.905565813769454,
     0.9153257715642311, 0.8834080717488789, 0.8728567660247956, 0.8696913743075706]

    F1 = [0.9594523433385993, 0.9479320037478249, 0.9517753047164812, 0.943844204348406, 0.9572967939337501,
     0.9512292358803986, 0.9583778014941302, 0.9583778014941302, 0.9492067724303427, 0.9464477933261572,
     0.9451855765420685, 0.9462623038042032, 0.9420174741858618, 0.9372595834991378, 0.9455818965517242,
     0.9305064782096584, 0.9299446931788253, 0.9243321489669695, 0.9401205626255861, 0.9233458848843464,
     0.9317937701396348, 0.9218276906138178, 0.9186563020544143, 0.8913219789132198]

    plt.plot(X, precision, color="lightcoral", linewidth=2, linestyle="-", label="precision")
    plt.plot(X, recall, color="burlywood", linewidth=2, linestyle="--", label="recall")
    plt.plot(X, F1, color="mediumturquoise", linewidth=2, linestyle="-.", label="F1")
    plt.legend(loc="best")
    plt.xlabel('most QP', fontsize=10)
    plt.ylabel('precenatge', fontsize=10)
    plt.show()


def twoQP_codingRate():
    X = [2*i for i in range(9, 25)]
    part = [655, 651, 640, 637, 631, 626, 619, 610, 605, 600, 608, 592, 599, 589, 579, 570]
    most = [1638, 1624, 1537, 1458, 1358, 1244, 1110, 1020, 925, 841, 777, 749, 704, 681, 658, 650]
    plt.plot(X, part, color="lightcoral", linewidth=2, linestyle="-", label="part")
    plt.plot(X, most, color="burlywood", linewidth=2, linestyle="--", label="most")
    plt.legend(loc="best")
    plt.xlabel('QP', fontsize=10)
    plt.ylabel('coding rate (kbps)', fontsize=10)
    plt.show()


def twoQP_F1():
    X = [2*i for i in range(1, 25)]
    part = [0.8913219789132198, 0.8927326360954678, 0.8940477797273586, 0.8851971907077256, 0.8812218737215328,
     0.8855168676331119, 0.8941271551724137, 0.8876786929884275, 0.8793478260869565, 0.886400431732326,
     0.8811570473461591, 0.8661353505335674, 0.8681558158539907, 0.8503915372990797, 0.8541553503530689,
     0.84755673117082, 0.8381962864721485, 0.812778817095985, 0.7672807663964628, 0.735944500650383, 0.6903981264637002,
     0.5733108108108107, 0.4694027116159766, 0.3853593779632088]
    most = [0.9594523433385993, 0.9479320037478249, 0.9517753047164812, 0.943844204348406, 0.9572967939337501,
     0.9512292358803986, 0.9583778014941302, 0.9583778014941302, 0.9492067724303427, 0.9464477933261572,
     0.9451855765420685, 0.9462623038042032, 0.9420174741858618, 0.9372595834991378, 0.9455818965517242,
     0.9305064782096584, 0.9299446931788253, 0.9243321489669695, 0.9401205626255861, 0.9233458848843464,
     0.9317937701396348, 0.9218276906138178, 0.9186563020544143, 0.8913219789132198]
    plt.plot(X, part, color="lightcoral", linewidth=2, linestyle="-", label="part")
    plt.plot(X, most, color="burlywood", linewidth=2, linestyle="--", label="most")
    plt.legend(loc="best")
    plt.xlabel('QP', fontsize=10)
    plt.ylabel('F1', fontsize=10)
    plt.show()


def twoQP_F1_bitrate():
    X = [2 * i for i in range(9, 25)]
    part_bitrate = [655, 651, 640, 637, 631, 626, 619, 610, 605, 600, 608, 592, 599, 589, 579, 570]
    most_bitrate = [1638, 1624, 1537, 1458, 1358, 1244, 1110, 1020, 925, 841, 777, 749, 704, 681, 658, 650]
    part_F1 = [0.8793478260869565, 0.886400431732326,
            0.8811570473461591, 0.8661353505335674, 0.8681558158539907, 0.8503915372990797, 0.8541553503530689,
            0.84755673117082, 0.8381962864721485, 0.812778817095985, 0.7672807663964628, 0.735944500650383,
            0.6903981264637002,
            0.5733108108108107, 0.4694027116159766, 0.3853593779632088]
    most_F1 = [0.9492067724303427, 0.9464477933261572, 0.9451855765420685, 0.9462623038042032, 0.9420174741858618,
               0.9372595834991378, 0.9455818965517242, 0.9305064782096584, 0.9299446931788253, 0.9243321489669695,
               0.9401205626255861, 0.9233458848843464, 0.9317937701396348, 0.9218276906138178, 0.9186563020544143,
               0.8913219789132198]

    font = {'family': 'serif',
            'serif': 'Times New Roman',
            'weight': 'normal',
            'size': 10}
    plt.rc('font', **font)
    ax1 = plt.gca()
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }

    ax1.tick_params(labelsize=14)
    plt.plot(X, part_bitrate, 'lightcoral', label="part bitrate", linewidth=2)
    plt.plot(X, most_bitrate, 'burlywood', label="most bitrate", linewidth=2)
    # 显示网格
    plt.grid(True)
    plt.xlabel("QP", font1)
    plt.ylabel('Bitrate(kbps)', font1)
    # plt.title("This is double axis label")
    # 设置线标的位置
    plt.legend(loc='center left', fontsize=14)

    # 第二纵轴的设置和绘图
    ax2 = ax1.twinx()
    plt.plot(X, part_F1, 'hotpink', label="part F1", linestyle='--', linewidth=2)
    plt.plot(X, most_F1, 'plum', label="most F1", linestyle='--', linewidth=2)
    plt.legend(loc='center right', fontsize=14)
    ax2.tick_params(labelsize=18)
    ax2.set_ylabel("Accuracy(F1 score)", font1)
    # 限制横轴显示刻度的范围
    # plt.xlim(0, 24)
    plt.show()


def region_qp_F1():
    X = [2*i for i in range(1, 25)]
    partF1 = [1.0, 0.9922061396532527, 0.9869385154507805, 0.9854748603351954, 0.9821117619123001, 0.9735534540791794,
     0.9740467798782442, 0.964337118183272, 0.9650416933932008, 0.9628566873904033, 0.9595245743655637,
     0.9560229445506693, 0.950431723696834, 0.9473684210526315, 0.9329083665338647, 0.9103448275862068,
     0.8960701526469634, 0.8786705388835109, 0.8711288711288712, 0.8088081090527788, 0.768579838116262,
     0.6794922649742167, 0.5990078544853246, 0.5382868937048504]
    mostF1 = [1.0, 0.9921996879875195, 0.9827856025039123, 0.9874411302982731, 0.9678972712680577, 0.9874804381846636,
     0.976303317535545, 0.9608313349320543, 0.964968152866242, 0.9709803921568628, 0.9637223974763408,
     0.9397024275646045, 0.8968192397207138, 0.9019011406844106, 0.9210526315789473, 0.859304084720121,
     0.8352769679300291, 0.8393250183418928, 0.893760539629005, 0.8252911813643927, 0.891846921797005,
     0.7938408896492729, 0.7448405253283302, 0.5584415584415585]
    F1 = [1.0, 0.9922050469018365, 0.9862361037586024, 0.9858071362249635, 0.979769932566442, 0.9759212451775975,
     0.9744272775705913, 0.9637526652452025, 0.9650293646556327, 0.9642289348171701, 0.9602348545503069,
     0.9532636038660134, 0.9412700517035664, 0.9395251213433031, 0.9309838472834068, 0.901241230437129,
     0.8849933598937584, 0.8715778336198914, 0.874860956618465, 0.8116695551704218, 0.7909008737571558,
     0.7010143294155531, 0.6253387533875339, 0.5423728813559322]
    plt.plot(X, partF1, color="lightcoral", linewidth=2, linestyle="-", label="part")
    plt.plot(X, mostF1, color="burlywood", linewidth=2, linestyle="--", label="most")
    plt.plot(X, F1, color="mediumturquoise", linewidth=2, linestyle="-.", label="all")
    plt.legend(loc="best")
    plt.xlabel('QP', fontsize=10)
    plt.ylabel('F1', fontsize=10)
    plt.show()


def qp_partSize():
    X = [2*i for i in range(1, 25)]
    Y1 = [2041, 1808, 1514, 1124, 912, 731, 593, 502, 421,
         345, 278, 226, 180, 143, 115, 92, 75, 61, 51, 41, 34, 28, 24, 19]
    Y2 = [13583, 11661, 9326, 7348, 5478, 4337, 3438, 2835, 2448, 2095, 1789, 1497, 1223, 990, 775, 611, 480, 380, 302,
         247, 201, 166, 140, 116]
    Y = []
    for i in range(24):
        Y.append(Y2[i]/Y1[i])
    plt.plot(
        X, Y, '-p', color='grey',
        marker='o',
        markersize=4, linewidth=2,
        # markerfacecolor='blue',
        markeredgecolor='mediumpurple',
        markeredgewidth=2)
    plt.xlabel('QP', fontsize=14)
    plt.ylabel('Part video coding rate (kbps)', fontsize=10)
    plt.show()


def all_qp():
    X = [18+2*i for i in range(16)]
    Y = [[1648.097, 1640.978, 1639.465, 1636.121, 1630.587, 1628.372, 1618.674, 1616.256,
      1615.894, 1610.045, 1606.567, 1606.4, 1606.103, 1606.073, 1603.356, 1597.84],
    [1617.819, 1612.961, 1609.465, 1606.377, 1605.12, 1599.395, 1596.395, 1590.903,
     1586.031, 1582.285, 1579.525, 1578.305, 1575.275, 1574.697, 1573.433, 1569.799],
    [1542.587, 1537.379, 1537.076, 1534.611, 1529.647, 1526.407, 1519.377, 1517.338,
      1509.115, 1503.767, 1501.824, 1504.741, 1503.338, 1499.581, 1498.387, 1491.546],
     [1463.657, 1460.912, 1456.226, 1455.24, 1454.994, 1453.296, 1447.609, 1438.523,
      1439.807, 1430.212, 1426.251, 1427.051, 1428.047, 1425.854, 1422.338, 1418.255],
    [1370.979, 1361.452, 1353.335, 1354.797, 1358.658, 1355.06, 1349.44, 1334.177,
     1342.53, 1329.531, 1326.595, 1337.558, 1329.482, 1327.391, 1321.241, 1324.928],
    [1238.101, 1240.928, 1235.783, 1236.268, 1226.653, 1224.237, 1214.902, 1214.356,
     1207.346, 1203.315, 1204.148, 1201.929, 1203.403, 1213.532, 1204.375, 1203.319],
    [1106.008, 1104.519, 1104.644, 1102.243, 1097.542, 1090.132, 1083.23, 1079.281,
     1078.17, 1073.873, 1075.847, 1072.748, 1074.275, 1074.422, 1075.756, 1069.183],
    [1020.37, 1020.952, 1012.094, 1016.891, 1015.824, 1011.909, 1002.963,  999.532,
     995.365, 993.255, 990.059, 990.007, 999.566, 990.549, 993.917, 983.961],
    [914.987, 915.488, 914.44, 909.205, 902.901, 903.786, 893.425, 887.761,
     879.531, 880.382, 879.974, 870.945, 878.168, 877.967, 877.051, 878.439],
    [861.536, 860.61,  857.798, 852.546, 855.927, 852.817, 847.347, 841.428,
     833.798, 835.009, 833.343, 827.857, 824.198, 830.363, 832.065, 826.548],
    [782.559, 783.467, 781.158, 777.622, 772.515, 770.208, 760.233, 755.343,
     752.406, 750.727, 744.751, 745.042, 753.519, 754.418, 747.232, 745.146],
    [749.837, 744.342, 743.048, 739.49,  733.466, 734.997, 722.661, 720.701,
     709.494, 710.411, 707.745, 708.333, 707.769, 709.836, 701.123, 705.67],
    [724.049, 722.201, 717.273, 712.913, 716.66, 710.585, 710.154, 705.171,
     690.466, 684.725, 684.19, 683.699, 685.572, 684.486, 674.931, 674.381],
    [693.478, 690.259, 689.077, 685.343, 678.809, 674.1,   662.211, 664.495,
     658.406, 655.535, 645.405, 642.911, 640.823, 642.553, 636.18,  634.804],
    [694.236, 690.725, 685.584, 679.801, 675.285, 672.524, 665.249, 660.777,
     656.664, 651.99, 645.951, 643.982, 641.681, 643.76, 636.3, 634.217],
    [654.897, 653.787, 645.386, 650.327, 647.769, 644.394, 632.319, 637.012,
     627.759, 623.398, 615.029, 607.862, 613.115, 614.341, 608.986, 612.93]]
    for i in range(16):
        plt.plot(X, Y[i], linewidth=1.0, linestyle="-", label=str(18+2*i))
    plt.legend(loc="best")
    plt.xlabel('part QP', fontsize=10)
    plt.ylabel('Coding rate (kbps)', fontsize=10)
    plt.show()


def qp_time_size_F1():
    time_1 = [5368, 5021, 4766, 4750, 4346, 4140, 4061, 3905]
    time_2 = [2647, 2484, 2363, 2305, 2300, 2216, 2104, 2132]
    size = [535.47, 412.28, 312.71, 262.50, 194.83, 151.78, 124.96, 108.87]
    F1 = [0.924,  0.924, 0.929, 0.914, 0.902, 0.897, 0.869, 0.837]
    plt.plot(time_1, size, color="lightcoral", linewidth=2, linestyle="-", label="1 core")
    plt.plot(time_2, size, color="burlywood", linewidth=2, linestyle="--", label="4 cores")
    plt.legend(loc="best")
    plt.xlabel('time(ms)', fontsize=10)
    plt.ylabel('size(kbps)', fontsize=10)
    plt.show()


def our():
    # size = [1/25*(10-i) for i in range(10)]
    #
    # h264 = [0.9386729717747895, 0.9345572993781462, 0.9234641006661731, 0.9183642673909762, 0.9196946564885496,
    #  0.9105373761633143, 0.8778025241276912, 0.8738920852122531, 0.7966831446340297, 0.6045123997762447]
    # plt.plot(size, h264, color="lightcoral", linewidth=2, linestyle="-", label="h.264")
    # plt.legend(loc="best")
    font = {'family': 'serif',
            'serif': 'Times New Roman',
            'weight': 'normal',
            'size': 14}
    plt.rc('font', **font)
    plt.xlabel('Norm. bitrate', fontsize=14)
    plt.ylabel('Accuracy(F1 score)', fontsize=14)
    plt.ylim(0.7, 1 + 1e-10)
    plt.xlim(0, 1)
    plt.yticks(np.arange(0.7, 1 + 1e-10, step=0.06))
    # fig = plt.figure(figsize=(6, 6))
    ellipses = [Ellipse(xy=(0.7444, 0.781),  # 中心，在 0~1 上均匀分布的 1 个 1*2 的数组
                        width=0.11699938177784212,  # 长半轴, 在 0~1 上均匀分布的 1 个 float
                        height=0.03846174169146105,  # 短半轴
                        angle=7.971753620536238),  # 旋转角度（逆时针）
                Ellipse((0.26948, 0.94), 0.09575200362089888, 0.01566546853618644, 4.696375781871432),
                Ellipse((0.3572, 0.91), 0.1658837109190766, 0.05063788333079787, -3.661425492794558)
                ]
    color = ["lightcoral", "burlywood", "mediumturquoise"]
    ax = plt.gca()
    for index, ellipse in enumerate(ellipses):
        ax.add_patch(p=ellipse)  # 向子区添加形状
        ellipse.set(
                    # alpha=0.7,
                    color=color[index] # 3 元 RGB 序列，在 0~1 上均匀分布的 1 个 1*3 的数组
                    )

    x = [0.7444, 0.26948, 0.3572]
    y = [0.781, 0.94, 0.91]
    n = ['DDS', 'Our', 'H.264']

    ax.scatter(x, y, marker='.', c='black', s=5)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))

    bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="gray", lw=1)
    t = ax.text(0.9, 0.73, "Better", ha="center", va="center", rotation=-45,
                size=10,
                bbox=bbox_props)
    bb = t.get_bbox_patch()
    bb.set_boxstyle("larrow", pad=0.6)  # 设置宽度

    plt.grid()  # 生成网格
    plt.show()


def dds():
    # size = [4397416, 3751145, 3111096, 2486536, 2240896, 1994401, 1748270]
    # h264 = [0.9668581528943879, 0.9572319572319572, 0.9402018376261485, 0.9259655014501603, 0.9120591406129679,
    #  0.8956671359299234, 0.8787396562698917]
    # plt.plot(size, h264, color="lightcoral", linewidth=2, linestyle="-", label="h.264")
    # plt.legend(loc="best")
    font = {'family': 'serif',
            'serif': 'Times New Roman',
            'weight': 'normal',
            'size': 14}
    plt.rc('font', **font)
    plt.xlabel('Norm. bitrate', fontsize=14)
    plt.ylabel('Accuracy(F1 score)', fontsize=14)
    plt.ylim(0.8, 1+1e-10)
    plt.xlim(0, 1)
    plt.yticks(np.arange(0.8, 1+1e-10, step=0.05))
    # fig = plt.figure(figsize=(6, 6))
    ellipses = [Ellipse(xy=(3471/4000, 0.901),  # 中心，在 0~1 上均匀分布的 1 个 1*2 的数组
                        width=0.1836004179392991,  # 长半轴
                        height=0.028322375615284303,  # 短半轴
                        angle=-3.613397512277034),  # 旋转角度（逆时针）
                Ellipse((1778/4000, 0.91), 0.08069243541095683, 0.029742564935326225, 2.5781025335916823),
                Ellipse((2488/4000, 0.912), 0.15450317772596733, 0.03624520920813925, -4.186949528137733)
                ]
    color = ["lightcoral", "burlywood", "mediumturquoise"]
    ax = plt.gca()
    for index, ellipse in enumerate(ellipses):
        ax.add_patch(p=ellipse)  # 向子区添加形状
        ellipse.set(
            # alpha=0.7,
            color=color[index]  # 3 元 RGB 序列，在 0~1 上均匀分布的 1 个 1*3 的数组
        )

    x = [3471/4000, 1778/4000, 2488/4000]
    y = [0.901, 0.91, 0.912]
    n = ['DDS', 'Our', 'H.264']
    ax = plt.gca()
    ax.scatter(x, y, marker='.', c='black', s=5)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))

    bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="gray", lw=1)
    t = ax.text(0.9, 0.825, "Better", ha="center", va="center", rotation=-45,
                size=10,
                bbox=bbox_props)
    bb = t.get_bbox_patch()
    bb.set_boxstyle("larrow", pad=0.6)  # 设置宽度

    plt.grid()  # 生成网格
    plt.show()


def high():
    font = {'family': 'serif',
            'serif': 'Times New Roman',
            'weight': 'normal',
            'size': 14}
    plt.rc('font', **font)
    plt.xlabel('Norm. bitrate', fontsize=14)
    plt.ylabel('Accuracy(F1 score)', fontsize=14)
    plt.ylim(0.75, 1 + 1e-10)
    plt.xlim(0, 1)
    plt.yticks(np.arange(0.75, 1 + 1e-10, step=0.05))
    # fig = plt.figure(figsize=(6, 6))
    ellipses = [Ellipse(xy=(10258/12000, 0.813),  # 中心，在 0~1 上均匀分布的 1 个 1*2 的数组
                        width=0.18675581851766634,  # 长半轴
                        height=0.021418342045506987,  # 短半轴
                        angle=3.8132992509121926),  # 旋转角度（逆时针）
                Ellipse((3174/12000, 0.91), 0.14407033224930585, 0.027764217223281058, 5.250188614535333),
                Ellipse((4587/12000, 0.8896), 0.19765939174147315, 0.03541445685088206, -1.6386297671064414)
                ]
    color = ["lightcoral", "burlywood", "mediumturquoise"]
    ax = plt.gca()
    for index, ellipse in enumerate(ellipses):
        ax.add_patch(p=ellipse)  # 向子区添加形状
        ellipse.set(
            # alpha=0.7,
            color=color[index]  # 3 元 RGB 序列，在 0~1 上均匀分布的 1 个 1*3 的数组
        )

    x = [10258/12000, 3174/12000, 4587/12000]
    y = [0.813, 0.91, 0.8896]
    n = ['DDS', 'Our', 'H.264']
    ax = plt.gca()
    ax.scatter(x, y, marker='.', c='black', s=5)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))

    bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="gray", lw=1)
    t = ax.text(0.9, 0.775, "Better", ha="center", va="center", rotation=-45,
                size=10,
                bbox=bbox_props)
    bb = t.get_bbox_patch()
    bb.set_boxstyle("larrow", pad=0.6)  # 设置宽度

    plt.grid()  # 生成网格
    plt.show()


def chunk_times():
    font = {'family': 'serif',
            'serif': 'Times New Roman',
            'weight': 'normal',
            'size': 14}
    plt.rc('font', **font)
    plt.xlabel('Norm. bitrate', fontsize=14)
    plt.ylabel('Accuracy(F1 score)', fontsize=14)
    plt.ylim(0.9, 1 + 1e-10)
    plt.xlim(0, 1)
    plt.yticks(np.arange(0.9, 1 + 1e-10, step=0.025))
    ellipses = [Ellipse(xy=(336.9/600, 0.9449),  # 中心，在 0~1 上均匀分布的 1 个 1*2 的数组
                        width=0.18042206112040873,  # 长半轴
                        height=0.011085118901647287,  # 短半轴
                        angle=2.2975947427669667),  # 旋转角度（逆时针）
                Ellipse((398.1/600, 0.9392), 0.1496446579749858, 0.010023745720684171, 0.29606357489774415),
                Ellipse((453.7/600, 0.9329), 0.16906929973351548, 0.012938481459661274, -1.0390672565102455),
                Ellipse((428.1/600, 0.917), 0.16453473770604907, 0.018233231728607156, -1.6456658649682905)
                ]
    color = ["lightcoral", "burlywood", "mediumturquoise", 'paleturquoise']
    ax = plt.gca()
    for index, ellipse in enumerate(ellipses):
        ax.add_patch(p=ellipse)  # 向子区添加形状
        ellipse.set(
            color=color[index]  # 3 元 RGB 序列，在 0~1 上均匀分布的 1 个 1*3 的数组
        )

    x = [336.9/600, 398.1/600, 453.7/600, 428.1/600]
    y = [0.9449, 0.9392, 0.9329, 0.917]
    n = ['Quartic', 'Thrice', 'Twice', 'Once']
    ax.scatter(x, y, marker='.', c='black', s=5)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))

    # 注释箭头
    bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="gray", lw=1)
    t = ax.text(0.9, 0.9125, "Better", ha="center", va="center", rotation=-45,
                size=10,
                bbox=bbox_props)
    bb = t.get_bbox_patch()
    bb.set_boxstyle("larrow", pad=0.6)  # 设置宽度

    plt.grid()  # 生成网格
    plt.show()


def chunk_time_bar():
    font = {'family': 'serif',
            'serif': 'Times New Roman',
            'weight': 'normal',
            'size': 14}
    plt.rc('font', **font)

    ax = plt.gca()

    times = ['Quartic', 'Thrice', 'Twice', 'Once']
    y_pos = np.arange(len(times))
    x = [336.9 / 1925, 398.1 / 1925, 453.7 / 1925, 428.1 / 1925]
    y = [0.9449, 0.9392, 0.9329, 0.917]
    # error = np.random.rand(len(people))

    total_width, n = 0.6, 2
    width = total_width / n
    y_pos = y_pos - (total_width - width) / 2

    b1 = ax.barh(y_pos, x, align='center',
                color='lightcoral', ecolor='black', height=0.2, label='Bandwidth (%)')
    ax.set_xlabel('Bandwidth (%)')
    ax.set_xlim(0.1, 0.35)
    # ax.legend(loc='upper right')

    ax2 = ax.twiny()
    b2 = ax2.barh(y_pos + width, y, align='center',
                color='mediumturquoise', ecolor='black', height=0.2, label='Accuracy (F1 score)')
    ax2.set_xlabel('Accuracy (F1 score)')
    ax2.set_xlim(0.8, 1)
    ax2.set_xticks(np.arange(0.8, 1 + 1e-10, step=0.05))
    # 添加数据标签
    for rect in b1:
        w = rect.get_width()
        ax.text(w, rect.get_y() + rect.get_height() / 2, '%.2f' % w, ha='left', va='center')
    for rect in b2:
        w = rect.get_width()
        ax2.text(w, rect.get_y() + rect.get_height() / 2, '%.2f' % w, ha='left', va='center')

    ax2.set_yticks(y_pos + width / 2.0)

    ax2.set_yticklabels(times)
    ax2.invert_yaxis()  # labels read top-to-bottom
    # ax2.legend(loc='upper right')

    plt.legend(handles=[b1, b2], labels=['Bandwidth (%)', 'Accuracy (F1 score)'], loc='best')
    plt.show()
    # print(y_pos + 3)


def h264_300():
    font = {'family': 'serif',
            'serif': 'Times New Roman',
            'weight': 'normal',
            'size': 14}
    plt.rc('font', **font)
    plt.xlabel('Encoder time (s)', fontsize=14)
    plt.ylabel('Accuracy(F1 score)', fontsize=14)
    plt.ylim(0.84, 1 + 1e-10)
    plt.xlim(3.0, 3.8)
    plt.yticks(np.arange(0.84, 1 + 1e-10, step=0.04))
    plt.xticks(np.arange(3, 3.8 + 1e-10, step=0.2))
    ellipses = [Ellipse(xy=(3.16, 0.9449),  # 中心，在 0~1 上均匀分布的 1 个 1*2 的数组
                        width=0.18042206112040873,  # 长半轴
                        height=0.011085118901647287,  # 短半轴
                        angle=2.2975947427669667),  # 旋转角度（逆时针）
                Ellipse((3.61, 0.869), 0.1105173673266501, 0.03619340649128404, 7.974508487213496),
                ]
    color = ["lightcoral", "burlywood", "mediumturquoise", 'paleturquoise']
    ax = plt.gca()
    for index, ellipse in enumerate(ellipses):
        ax.add_patch(p=ellipse)  # 向子区添加形状
        ellipse.set(
            color=color[index]  # 3 元 RGB 序列，在 0~1 上均匀分布的 1 个 1*3 的数组
        )

    x = [3.16, 3.61]
    y = [0.9449, 0.869]
    n = ['Our', 'H.264']
    ax.scatter(x, y, marker='.', c='black', s=5)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))

    # 注释箭头
    bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="gray", lw=1)
    t = ax.text(3.7, 0.855, "Better", ha="center", va="center", rotation=-45,
                size=9,
                bbox=bbox_props)
    bb = t.get_bbox_patch()
    bb.set_boxstyle("larrow", pad=0.6)  # 设置宽度

    plt.grid()  # 生成网格
    plt.show()


def h264_15():
    font = {'family': 'serif',
            'serif': 'Times New Roman',
            'weight': 'normal',
            'size': 14}
    plt.rc('font', **font)
    plt.xlabel('Encoder time (s)', fontsize=14)
    plt.ylabel('Accuracy(F1 score)', fontsize=14)
    plt.ylim(0.75, 0.95 + 1e-10)
    plt.xlim(0.1, 0.3)
    plt.yticks(np.arange(0.75, 0.95 + 1e-10, step=0.05))
    plt.xticks(np.arange(0.1, 0.3 + 1e-10, step=0.05))
    ellipses = [Ellipse(xy=(0.23, 0.8857),  # 中心，在 0~1 上均匀分布的 1 个 1*2 的数组
                        width=0.10000000000000002,  # 长半轴
                        height=0.05,  # 短半轴
                        angle=50.49129176518768),  # 旋转角度（逆时针）
                Ellipse((0.21, 0.7925), 0.08, 0.04, 26.78531065117421),
                ]
    color = ["lightcoral", "burlywood", "mediumturquoise", 'paleturquoise']
    ax = plt.gca()
    for index, ellipse in enumerate(ellipses):
        ax.add_patch(p=ellipse)  # 向子区添加形状
        ellipse.set(
            color=color[index]  # 3 元 RGB 序列，在 0~1 上均匀分布的 1 个 1*3 的数组
        )

    x = [0.23, 0.21]
    y = [0.8857, 0.7925]
    n = ['Our', 'H.264']
    ax.scatter(x, y, marker='.', c='black', s=5)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))

    # 注释箭头
    bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="gray", lw=1)
    t = ax.text(0.275, 0.775, "Better", ha="center", va="center", rotation=-45,
                size=10,
                bbox=bbox_props)
    bb = t.get_bbox_patch()
    bb.set_boxstyle("larrow", pad=0.6)  # 设置宽度

    plt.grid()  # 生成网格
    plt.show()


def pipline():
    X = [i for i in range(1, 21)]
    accuracy = [0.849, 0.925, 0.789, 0.883, 0.861, 0.881, 0.884, 0.912, 0.869, 0.786, 0.856, 0.861, 0.921, 0.890,
                0.911, 0.901, 0.773, 0.812, 0.809, 0.899]
    plt.plot(X, accuracy, color="lightcoral", linewidth=2)
    # plt.legend(loc="best")
    plt.xlabel('time slot', fontsize=10)
    plt.ylabel('accuracy(F1 score)', fontsize=10)
    # plt.ylim(0.75, 0.95 + 1e-10)
    # plt.xlim(0.1, 0.3)
    plt.yticks(np.arange(0.75, 0.95 + 1e-10, step=0.05))
    plt.xticks(np.arange(0, 20 + 1e-10, step=4))
    plt.show()


if __name__ == '__main__':
    # resolution_F1()
    # qp_F1()
    # QP_codeRate()
    # codeRate_F1()
    # partQP_codingRate()
    # partQP_F1()
    # mostQP_codingRate()
    # mostQP_F1()
    # twoQP_codingRate()
    # twoQP_F1()
    # twoQP_F1_bitrate()
    # region_qp_F1()
    # qp_partSize()
    # all_qp()
    # qp_time_size_F1()
    # qp_codeRate_F1()
    # our()
    # dds()
    # high()
    # chunk_times()
    chunk_time_bar()
    # h264_300()
    # h264_15()
    # pipline()
