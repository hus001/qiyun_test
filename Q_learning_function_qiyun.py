# 导入相关的第三方包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

## 忽略提醒
import warnings
warnings.filterwarnings("ignore")


def get_reward_values(df,stage_index):
    
    # R = 0
    if df.loc[stage_index,'stimulus']==1:     # GO_cue
        
        if df.loc[stage_index,'outcome'] ==1:      # Hit
            R = 1.0
        elif df.loc[stage_index,'outcome'] == 2:   # Miss
            R = 0
        elif df.loc[stage_index,'outcome'] == 3:   # CR
            R = 0
        elif df.loc[stage_index,'outcome'] == 4:   # FA
            R = -1.0
        elif df.loc[stage_index,'outcome'] == 5:   # RO
            R = 0

    elif df.loc[stage_index,'stimulus']==0:     # No_GO_cue
        
        if df.loc[stage_index,'outcome'] ==1:      # Hit
            R = 1.0
        elif df.loc[stage_index,'outcome'] == 2:   # Miss
            R = 0
        elif df.loc[stage_index,'outcome'] == 3:   # CR
            R = 0
        elif df.loc[stage_index,'outcome'] == 4:   # FA
            R = -1.0
        elif df.loc[stage_index,'outcome'] == 5:   # RO
            R = 0
    
    return R


def get_two_Q_list(df_go,df_nogo,stagename):
    '''
    df: dataframe 传入数据行为每一次试验,column0:每次试验所给的信号,column1:为每次输出的结果或者动作.
    N_STATES: 所切换的状态的总数,如试验的总次数.
    ALPHA: float 取值为0~1,学习率
    GAMMA: float 取值为0~1,折扣率
    stage_name: 获取Q值的阶段,如stable,uncertain,reverse
    return Q_go,Q_nogo: 返回单步更新的 Q_go,Q_nogo 值
    # 算法原理（贝尔曼方程）
    # Q(s, a) = Q(s, a) + α(r + γ * Q(s', a') - Q(s, a))
    '''

    Q_go = []                         # 创建存储迭代过程的go_Q值空列表
    Q_nogo= []                        # 创建存储迭代过程的nogo_Q值空列表

    # 初始化迭代初值
    if stagename == 'stable':
        Qs_go = 1.1
        Qs_nogo = 0.1
        ALPHA = 0.1
        GAMMA = 0.1
    elif stagename == 'uncertain':
        Qs_go = 1.1
        Qs_nogo = 0.1
        ALPHA = 0.1
        GAMMA = 0.1

    elif stagename == 'reverse':
        Qs_go = 1.1
        Qs_nogo = 0.1
        ALPHA = 0.5
        GAMMA = 0.1


    Q_go.append(Qs_go)
    Q_nogo.append(Qs_nogo)

    go_N_STATES = np.array(df_go).shape[0]
    nogo_N_STATES = np.array(df_nogo).shape[0]

    for i in range(go_N_STATES-1):
        
        R = get_reward_values(df_go,i)
        Qs_go_next_R = get_reward_values(df_go,i+1)
        Qs_go = Qs_go + ALPHA*(R + GAMMA * Qs_go_next_R - Qs_go)
        Q_go.append(Qs_go)
        # if i == go_N_STATES-2:
        #     Q_go.append(Qs_go)


    for j in range(nogo_N_STATES-1):

        R = get_reward_values(df_nogo,j)
        Qs_nogo_next_R = get_reward_values(df_nogo,j+1)
        Qs_nogo = Qs_nogo + ALPHA*(R + GAMMA * Qs_nogo_next_R - Qs_nogo)
        Q_nogo.append(Qs_nogo)

        # if j == nogo_N_STATES-2:
        #     Q_nogo.append(Qs_nogo)

    return Q_go,Q_nogo,Q_go[0],Q_nogo[0],ALPHA,GAMMA


def get_file_list(file_path):
    '''
    file_path:str        # 文件夹路径
    return:list          # 返回对应文件夹下所有对应格式组成的路径列表
    '''
    file_list = glob.glob(os.path.join(file_path,'*csv'))
    
    return file_list


def scatter_plot(df1,df2,mice_num):
    '''
    # 绘制两行一列的子图函数
    df1:dataframe
    df2:dataframe
    mice_num:str
    对小鼠的stable_uncertain.stable_reverse 两个阶段进行绘图
    '''

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    label_dict = {1:'Hit',2:'Miss',3:'CR',4:'FA',5:'RO',6:'UR'}
    for label, color in zip(range(1, 7), ('blue', 'red', 'green', 'yellow', 'black','orange')):

        # 绘制第一个子图
        axs[0].scatter(df1.index[df1['outcome']==label], df1['Q_values'][df1['outcome']==label],label = label_dict[label])
        axs[0].set_title(f'mice{mice_num}_stable_uncertain_Q_vari')
        axs[0].legend(loc='upper right')
        # 绘制第二个子图
        axs[1].scatter(df2.index[df2['outcome']==label], df2['Q_values'][df2['outcome']==label],label = label_dict[label])
        axs[1].set_title(f'mice{mice_num}_stable_reverse_Q_vari')
        axs[1].legend(loc='upper right')

    # 设置x轴和y轴标签
    for ax in axs.flat:
        ax.set(xlabel='Index', ylabel='Q_values')

    # 显示图形
    plt.show()


# 绘制单阶段的子图函数
def scatter1_plot(df1,mice_num,stage_name,Q0,ALPHA,GAMMA):
    '''
    df1:dataframe
    df2:dataframe
    mice_num:str
    对小鼠的stable_uncertain.stable_reverse 两个阶段进行绘图
    '''

    fig, axs = plt.subplots(1, 1, figsize=(10, 3))
    label_dict = {1:'Hit',2:'Miss',3:'CR',4:'FA',5:'RO',6:'UR'}
    for label, color in zip(range(1, 7), ('blue', 'red', 'green', 'yellow', 'black','orange')):

        # 绘制第一个子图
        plt.scatter(df1.index[df1['outcome']==label], df1['Q_values'][df1['outcome']==label],label = label_dict[label])
        plt.title(f'mice{mice_num}_{stage_name}_Q_vari')
        plt.legend(loc='upper right')

    # 设置x轴和y轴标签
    plt.xlabel('Index(trail by trail)') 
    plt.ylabel('Q_values')
    expression = f'Q0:{Q0}\nALPHA:{ALPHA}\nGAMMA:{GAMMA}'
    plt.annotate(expression, xy=(0.2, 0.2), xycoords='axes fraction', fontsize=12, ha='center')
    plt.grid()
    # 显示图形
    plt.show()


def get_mice_goNogo_data():
    '''获取每只小鼠三个阶段go/nogo的行为学数据
    return mice_go_three_stage_data mice_nogo_three_stage_data
    可通过列表访问的方式访问每一只老鼠
    '''
    col_name = ['stable','uncertain','reverse']

    stage_go= []               # 3X6 三个阶段六只小鼠
    stage_nogo= []             # 3X6 三个阶段六只小鼠
    for m in col_name:

        file_list = get_file_list(r'E:\project_wuqiyun\Pyddm_wuqiyun\原始数据\{}'.format(m))

        mice_go = []
        mice_nogo = []

        for i in range(len(file_list)):

            df = pd.read_csv(file_list[i])
            df = df.loc[:,['stimulus','outcome']]

            # 将数据根据go/nogo分为两组
            df_go = df[df['stimulus']==1]
            df_nogo = df[df['stimulus']==0]
            df_go.reset_index(inplace=True)
            df_go.drop(columns='index',axis=1,inplace=True)
            df_nogo.reset_index(inplace=True)
            df_nogo.drop(columns='index',axis=1,inplace=True)
            # if m == 'reverse':
            #     df_go = df[df['stimulus']==0]
            #     df_nogo = df[df['stimulus']==1]
            #     df_go.reset_index(inplace=True)
            #     df_go.drop(columns='index',axis=1,inplace=True)
            #     df_nogo.reset_index(inplace=True)
            #     df_nogo.drop(columns='index',axis=1,inplace=True)
            # else:
            # # 将数据根据go/nogo分为两组
            #     df_go = df[df['stimulus']==1]
            #     df_nogo = df[df['stimulus']==0]
            #     df_go.reset_index(inplace=True)
            #     df_go.drop(columns='index',axis=1,inplace=True)
            #     df_nogo.reset_index(inplace=True)
            #     df_nogo.drop(columns='index',axis=1,inplace=True)
            stable_Q_go,stable_Q_nogo,Qs_go,Qs_nogo,ALPHA,GAMMA = get_two_Q_list(df_go,df_nogo,m)
            
            df_go['Q_values'] = stable_Q_go
            df_nogo['Q_values'] = stable_Q_nogo

            df_go['stage'] = m
            df_nogo['stage'] = m
            
            df_go['subject'] = i+1
            df_nogo['subject'] = i+1
            
            mice_go.append(df_go)
            mice_nogo.append(df_nogo)
        stage_go.append(mice_go)
        stage_nogo.append(mice_nogo)

    # 对stage_go and stage_nogo进行拼接，生成每只小鼠在三个阶段go、nogo的数据  stage_go[m][n]  m:阶段编号，n:小鼠编号
    mice_go_three_stage_data = []    
    for mn_go in range(5):
        mice_df = pd.concat([stage_go[0][mn_go],stage_go[1][mn_go],stage_go[2][mn_go]],axis=0)
        mice_go_three_stage_data.append(mice_df)

    mice_nogo_three_stage_data = []      
    for mn_nogo in range(5):
        mice_df = pd.concat([stage_go[0][mn_nogo],stage_go[1][mn_nogo],stage_go[2][mn_nogo]],axis=0)
        mice_nogo_three_stage_data.append(mice_df)

    return mice_go_three_stage_data, mice_nogo_three_stage_data


def calculate_probability(sub_df):
    
    total_rows = len(sub_df)
    outcome_1_rows = len(sub_df[sub_df['outcome'] == 1]) + len(sub_df[sub_df['outcome'] == 4]) + len(sub_df[sub_df['outcome'] == 5])
    probability = outcome_1_rows / total_rows

    return probability


def calculate_Q_values(sub_df,calculate_mode):
    '''
    滑动窗口或者子阶段dataframe的Q值计算
    sub_df: 子阶段dataframe 需要包含Q_values这一列
    calculate_mode : 定义Q值计算方式,0:均值计计算,1:表示采用子阶段的第一个值
    return:Q_values
    '''
    if calculate_mode == 0  :                                  # 选择滑动窗口sub_dataframe Q_values.mean()作为Q值大
        return sub_df['Q_values'].mean()
    elif calculate_mode == 1:                                  # 选择滑动窗口sub_dataframe Q_values.first()作为Q值大小
        return np.array(sub_df['Q_values'])[0]
    

# 定义拟合函数
def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-(a * x + b)))


def slide_window_probabilities(df,windowSize,stepSize,calculate_Q_mode):

    window_size = windowSize  # 滑动窗口大小
    step_size = stepSize  # 步长

    probabilities = []  # 存储计算得到的概率值
    slide_Q_value = []  # 存储slide_window计算得到的Q值
    for i in range(0, len(df) - window_size + 1, step_size):
        sub_df = df.iloc[i:i+window_size]
        probability = calculate_probability(sub_df)
        probabilities.append(probability)
        slide_Q = calculate_Q_values(sub_df,calculate_Q_mode)
        slide_Q_value.append(slide_Q)

    return probabilities,slide_Q_value
    # # 执行拟合
    # # x = np.arange(len(probabilities))
    # x = slide_Q_value
    # # popt, pcov = curve_fit(sigmoid, x, probabilities)

    # # a = popt[0]  # 拟合参数 a
    # # b = popt[1]  # 拟合参数 b
    # # expression = f'y = 1 / (1 + exp(-({a} * x + {b})))'
    # # print(expression)

    # # 绘制原始数据和拟合曲线
    # plt.plot(x, probabilities, 'bo', label='Original Data')
    # # plt.plot(x, sigmoid(x, *popt), 'r-', label='Fitted Curve')
    # plt.xlabel('slide_Q_value')
    # plt.ylabel('Probability')
    # plt.title(f'mice{mice_number}_Go_{stage_name}_HitProbabilities')
    # # expression = r'$y = \frac{1}{{1 + e^{{-(%.2f \cdot x + %.2f)}}}}$' % (a, b)
    # # plt.annotate(expression, xy=(0.2, 0.2), xycoords='axes fraction', fontsize=12, ha='center')
    # plt.grid()
    # plt.legend()
    # plt.show()


def CDF_window_probabilities(df,stepSize,calculate_Q_mode):
    '''
    类似累积分布函数中积分的作用求概率值
    '''
    step_size = stepSize  # 步长
    totol_step = len(df)//step_size
    probabilities = []  # 存储计算得到的概率值
    slide_Q_value = []  # 存储slide_window计算得到的Q值
    for i in range(1, totol_step):
        sub_df = df.iloc[0:i*step_size]
        probability = calculate_probability(sub_df)
        probabilities.append(probability)
        slide_Q = calculate_Q_values(sub_df,calculate_Q_mode)
        slide_Q_value.append(slide_Q)

    return probabilities,slide_Q_value


    
# 定义三阶段绘图函数
def probabilities_slide_Q_value_plot(slide_Q_value1,slide_Q_value2,slide_Q_value3,probabilities1,probabilities2,probabilities3,mice_number,plot_mode,window_size,step_size):  
    '''
    根据三个阶段 Q 值或者slide_index 绘制散点图
    plot_mode:0  以Q VS P
    plot_mode:1  以slideIndex VS P
    '''
    # 执行拟合
    # x = np.arange(len(probabilities))
    if plot_mode == 0 :
        x1 = slide_Q_value1
        x2 = slide_Q_value2
        x3 = slide_Q_value3
    elif plot_mode == 1:
        x1 = np.arange(len(probabilities1))
        x2 = np.arange(len(probabilities2))
        x3 = np.arange(len(probabilities3))

    P1 = probabilities1
    P2 = probabilities2
    P3 = probabilities3

    # popt, pcov = curve_fit(sigmoid, x, probabilities)
    # a = popt[0]  # 拟合参数 a
    # b = popt[1]  # 拟合参数 b
    # expression = f'y = 1 / (1 + exp(-({a} * x + {b})))'
    # print(expression)
    # 绘制原始数据和拟合曲线
    plt.plot(x1, P1, 'ro', label='stable')
    plt.plot(x2, P2, 'bo', label='uncertain')
    plt.plot(x3, P3, 'go', label='reverse')
    # plt.plot(x, sigmoid(x, *popt), 'r-', label='Fitted Curve')
    if plot_mode == 0 :
        plt.xlabel('slide_Q_value')
        expression = f'window_size:{window_size}\nstep_size:{step_size}'
        plt.annotate(expression, xy=(0.8, 0.2), xycoords='axes fraction', fontsize=12, ha='center')
    else:
        plt.xlabel('slide_window_index')
        expression = f'step_size:{step_size}'
        plt.annotate(expression, xy=(0.8, 0.2), xycoords='axes fraction', fontsize=12, ha='center')

    plt.ylabel('Probability')
    plt.title(f'mice{mice_number+1}_Go_LickProbabilities')
    # expression = r'$y = \frac{1}{{1 + e^{{-(%.2f \cdot x + %.2f)}}}}$' % (a, b)
    # plt.annotate(expression, xy=(0.2, 0.2), xycoords='axes fraction', fontsize=12, ha='center')
    plt.grid()
    plt.legend()
    plt.show()
    

# =========================自定义的QQP转换函数===========================================================
def qiyun_softmax_update(x, scale_factor=3.0):
    """Softmax function with linear scaling factor"""
    e_x = np.exp(scale_factor * (x - np.max(x)))
    return e_x / e_x.sum()

    # return e_x / e_x.sum(axis=1, keepdims=True)


def qiyun_softmax_update2(x, scale_factor=1):
    """Softmax function with linear scaling factor"""
    e_x = np.exp(scale_factor * (x - np.max(x)))
    probabilities = e_x / e_x.sum(axis=1, keepdims=True)
    scaled_probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    return scaled_probabilities

def qiyun_softmax_update3(x, scale_factor=2):
    """Softmax function with power scaling factor"""
    scaled_x = np.power(x, scale_factor)
    e_x = np.exp(scaled_x - np.max(scaled_x))
    probabilities = e_x / e_x.sum(axis=1, keepdims=True)
    return probabilities


def qiyun_sigmoid_update(x, scale_factor=1):
    """Sigmoid function with linear scaling factor"""
    sigmoid_x = 1 / (1 + np.exp(-scale_factor * (x - np.max(x))))
    return sigmoid_x / sigmoid_x.sum()


def qiyun_sigmoid_update2(x, scale_factor=3):
    """Sigmoid function with linear scaling factor"""
    scaled_x = scale_factor * (x - np.mean(x))  # 缩放Q值，可根据需求调整
    probabilities = 1 / (1 + np.exp(-scaled_x))
    normalized_probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    return normalized_probabilities



def qiyun_gaussian_update(x, mean, std):
    """Gaussian function with mean and standard deviation"""
    exponent = -((x - mean)**2) / (2 * std**2)
    gaussian_x = np.exp(exponent) / (std * np.sqrt(2 * np.pi))
    return gaussian_x / gaussian_x.sum()


# def softmax(Q):
#     exp_values = np.exp(Q*2.4)
#     P_sum = exp_values[:,0]+exp_values[:,1]
#     P_lick = exp_values[:,0] / P_sum
#     P_Nolick = exp_values[:,1] / P_sum
#     P = np.array([P_lick,P_Nolick])
#     # probabilities = exp_values / np.sum(exp_values,axis=0)
#     # print(probabilities.shape)
#     return P


def qiyun_softmax(x):
    """Softmax function to convert Q-values to probabilities"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ==========================================================================================
# def generate_action(init_lick_Q,init_Nolick_Q,action_len,Alpha,Gamma):

#     '''
#     init_lick_Q :lick 初始 Q 值
#     init_Nolick_Q :nolick 初始 Q 值
#     action_len: 模拟生成的行动序列长度
#     '''
#     # Set initial Q-values
#     lick_q = init_lick_Q
#     nolick_q = init_Nolick_Q

#     # Iteration count
#     N = action_len

#     # Learning rate and discount factor
#     alpha = Alpha
#     gamma = Gamma

#     # Create an empty DataFrame to record data
#     columns = ['Lick Q', 'NoLick Q', 'Lick Probability', 'NoLick Probability', 'Action']
#     data = pd.DataFrame(columns=columns)

#     # Iterate to generate data
#     for i in range(N):
#         # Convert Q-values to probabilities
#         # lick_prob = qiyun_softmax_update2([lick_q, nolick_q])[0]
#         # nolick_prob = qiyun_softmax_update2([lick_q, nolick_q])[1]
# # =====================================================================================
#         Q = np.array([[lick_q, nolick_q]])
#         probs = qiyun_softmax_update(Q)
#         lick_prob = probs[0, 0]
#         nolick_prob = probs[0, 1]
#         # print(lick_prob,nolick_prob)
# # =========================================================================
#         # print(lick_prob)
#         # Choose action based on probabilities
#         action = np.random.choice(['Lick', 'NoLick'], p=[lick_prob, nolick_prob])
#         # 下一次预期的奖励
#         action_next = np.random.choice(['Lick', 'NoLick'], p=[lick_prob, nolick_prob])
#         if action_next == 'Lick':
#             # 10% probability of setting reward to 0
#             if np.random.random() <= 0.1: 
#                 action_next = 'RO'
#                 next_reward = 0
#             else: next_reward = 1
#         elif action_next == 'NoLick':
#             next_reward = 0
#         # ==========================================================================
#         # 当前动作的奖励
#         if action == 'Lick':
#             # 10% probability of setting reward to 0
#             if np.random.random() <= 0.1: 
#                 action = 'RO'
#                 reward = 0
#             else: reward = 1
            
#             # Record data
#             data.loc[i] = [lick_q, nolick_q, lick_prob, nolick_prob, action]
            
#             # Update Q-values using learning rate and discount factor
#             new_lick_q = lick_q + alpha * (reward + gamma * next_reward - lick_q)
#             # new_nolick_q = nolick_q + alpha * (0 + gamma * (1*nolick_prob)- nolick_q)
#             new_nolick_q = nolick_q 


#             # new_lick_q = lick_q + alpha * (reward + gamma * (lick_prob)) - lick_q)
#             # new_nolick_q = nolick_q + alpha * (0 + gamma * nolick_q - nolick_q)

#             lick_q, nolick_q = new_lick_q, new_nolick_q
            
#         elif action == 'NoLick':
#             reward = 0
            
#             # Record data
#             data.loc[i] = [lick_q, nolick_q, lick_prob, nolick_prob, action]
            
#             # Update Q-values using learning rate and discount factor
#             # new_lick_q = lick_q + alpha * (0 + gamma * (1*lick_prob )- lick_q)
#             new_lick_q = lick_q 
#             new_nolick_q = nolick_q + alpha * (0 + gamma * next_reward - nolick_q)

#             lick_q, nolick_q = new_lick_q, new_nolick_q
            
#     return data


def generate_action(init_lick_Q,init_Nolick_Q,action_len,Alpha,Gamma):

    '''
    更新迭代公式后的动作序列生成函数
    init_lick_Q :lick 初始 Q 值
    init_Nolick_Q :nolick 初始 Q 值
    action_len: 模拟生成的行动序列长度
    '''
    # Set initial Q-values
    lick_q = init_lick_Q
    nolick_q = init_Nolick_Q

    # Iteration count
    N = action_len

    # Learning rate and discount factor
    alpha = Alpha
    THETA = 0.08
    THETA_p = 0.6
    PE = 0
    gamma = Gamma

    # Create an empty DataFrame to record data
    columns = ['Lick Q', 'NoLick Q', 'Lick Probability', 'NoLick Probability', 'Action', 'Alpha', 'PE']
    data = pd.DataFrame(columns=columns)

    # Iterate to generate data
    for i in range(N):
        # Convert Q-values to probabilities
        # lick_prob = qiyun_softmax_update2([lick_q, nolick_q])[0]
        # nolick_prob = qiyun_softmax_update2([lick_q, nolick_q])[1]
# =====================================================================================
        Q = np.array([[lick_q, nolick_q]])
        probs = qiyun_softmax_update(Q)
        lick_prob = probs[0, 0]
        nolick_prob = probs[0, 1]
        # print(lick_prob,nolick_prob)
# =========================================================================
        # print(lick_prob)
        # Choose action based on probabilities
        action = np.random.choice(['Lick', 'NoLick'], p=[lick_prob, nolick_prob])
        # 当前动作的奖励
        if action == 'Lick':
            # 10% probability of setting reward to 0
            if np.random.random() <= 0.2: 
                action = 'RO'
                reward = -0.1
            else: reward = 1
            
            # Record data
            data.loc[i] = [lick_q, nolick_q, lick_prob, nolick_prob, action, alpha, PE]
            
            # Update Q-values using learning rate and discount factor
            PE= reward - lick_q
            alpha = THETA_p*alpha+THETA*abs(PE)

            new_lick_q = lick_q + alpha * (reward - lick_q)
            new_nolick_q = nolick_q 

            lick_q, nolick_q = new_lick_q, new_nolick_q
            
        elif action == 'NoLick':
            reward = 0.1
        
            # Record data
            data.loc[i] = [lick_q, nolick_q, lick_prob, nolick_prob, action, alpha, PE]
            
            # Update Q-values using learning rate and discount factor
            PE= reward - nolick_q
            alpha = THETA_p*alpha+THETA*abs(PE)

            new_lick_q = lick_q 
            new_nolick_q = nolick_q + alpha * (reward - nolick_q)

            lick_q, nolick_q = new_lick_q, new_nolick_q
            
    return data


def generate_action_uncertain_update(init_lick_Q,init_Nolick_Q,action_len,Alpha,Gamma):

    '''
    init_lick_Q :lick 初始 Q 值
    init_Nolick_Q :nolick 初始 Q 值
    action_len: 模拟生成的行动序列长度
    '''
    # Set initial Q-values
    lick_q = init_lick_Q
    nolick_q = init_Nolick_Q

    # Iteration count
    N = action_len

    # Learning rate and discount factor
    alpha = Alpha
    gamma = Gamma

    # Create an empty DataFrame to record data
    columns = ['Lick Q', 'NoLick Q', 'Lick Probability', 'NoLick Probability', 'Action']
    data = pd.DataFrame(columns=columns)

    # Iterate to generate data
    for i in range(N):
        # Convert Q-values to probabilities
        lick_prob = qiyun_sigmoid_update([lick_q, nolick_q])[0]
        nolick_prob = qiyun_sigmoid_update([lick_q, nolick_q])[1]

        # print(lick_prob)
        # Choose action based on probabilities
        action = np.random.choice(['Lick', 'NoLick'], p=[lick_prob, nolick_prob])
               # 当前动作的奖励
        if action == 'Lick':
            # 10% probability of setting reward to 0
            if np.random.random() <= 0.1: 
                action = 'RO'
                reward = 0
            else: reward = 1
            
            lick_q_0 = lick_q + alpha * (reward - lick_q)
            nolick_q_0 = nolick_q 

        elif action == 'NoLick':
            reward = 0
            
            lick_q_0 = lick_q 
            nolick_q_0 = nolick_q + alpha * (0 - nolick_q)
        # 通过当前动作及时更新Q值，以便于做出下一次动作的预期近似
        lick_prob_0 = qiyun_softmax_update([lick_q_0, nolick_q_0])[0]
        nolick_prob_0 = qiyun_softmax_update([lick_q_0, nolick_q_0])[1]

        action_next = np.random.choice(['Lick', 'NoLick'], p=[lick_prob_0, nolick_prob_0])
        # 下一次预期的奖励
        if action_next == 'Lick':
            # 10% probability of setting reward to 0
            if np.random.random() <= 0.1: 
                action_next = 'RO'
                next_reward = 0
            else: next_reward = 1
        elif action_next == 'NoLick':
            next_reward = 0

        # ==========================================================================
        # 当前动作的奖励
        if action == 'Lick':
            # 10% probability of setting reward to 0
            if np.random.random() <= 0.1: 
                action = 'RO'
                reward = 0
            else: reward = 1
            
            # Record data
            data.loc[i] = [lick_q, nolick_q, lick_prob, nolick_prob, action]
            
            # Update Q-values using learning rate and discount factor
            new_lick_q = lick_q + alpha * (reward + gamma * next_reward - lick_q)
            new_nolick_q = nolick_q 

            lick_q, nolick_q = new_lick_q, new_nolick_q
            
        elif action == 'NoLick':
            reward = 0
            
            # Record data
            data.loc[i] = [lick_q, nolick_q, lick_prob, nolick_prob, action]
            
            new_lick_q = lick_q 
            new_nolick_q = nolick_q + alpha * (0 + gamma * next_reward - nolick_q)

            lick_q, nolick_q = new_lick_q, new_nolick_q
            
    return data


# def generate_action_reverse(init_lick_Q,init_Nolick_Q,action_len,Alpha,Gamma):

#     '''
#     init_lick_Q :lick 初始 Q 值
#     init_Nolick_Q :nolick 初始 Q 值
#     action_len: 模拟生成的行动序列长度
#     '''
#     # Set initial Q-values
#     lick_q = init_lick_Q
#     nolick_q = init_Nolick_Q

#     # Iteration count
#     N = action_len

#     # Learning rate and discount factor
#     alpha = Alpha
#     gamma = Gamma

#     # Create an empty DataFrame to record data
#     columns = ['Lick Q', 'NoLick Q', 'Lick Probability', 'NoLick Probability', 'Action']
#     data = pd.DataFrame(columns=columns)

#     # Iterate to generate data
#     for i in range(N):
#         # Convert Q-values to probabilities
#         # lick_prob = qiyun_softmax_update([lick_q, nolick_q])[0]
#         # nolick_prob = qiyun_softmax_update([lick_q, nolick_q])[1]
#         # ============================================================
#         Q = np.array([[lick_q, nolick_q]])
#         probs = qiyun_softmax_update(Q)
#         lick_prob = probs[0, 0]
#         nolick_prob = probs[0, 1]
#         # print(lick_prob,nolick_prob)
#         # ============================================================
#         # print(lick_prob)
#         # Choose action based on probabilities
#         action = np.random.choice(['Lick', 'NoLick'], p=[lick_prob, nolick_prob])
#         # 下一次预期的奖励
#         action_next = np.random.choice(['Lick', 'NoLick'], p=[lick_prob, nolick_prob])
#         if action_next == 'Lick':
#             next_reward = -1
#         elif action_next == 'NoLick':
#             next_reward = 0
#         # ==========================================================================
#         # 当前动作的奖励
#         if action == 'Lick':
#             reward = -1
            
#             # Record data
#             data.loc[i] = [lick_q, nolick_q, lick_prob, nolick_prob, action]
            
#             # Update Q-values using learning rate and discount factor
#             new_lick_q = lick_q + alpha * (reward + gamma * next_reward - lick_q)
#             # new_nolick_q = nolick_q + alpha * (0 + gamma * (1*nolick_prob)- nolick_q)
#             new_nolick_q = nolick_q 

#             # new_lick_q = lick_q + alpha * (reward + gamma * (lick_prob)) - lick_q)
#             # new_nolick_q = nolick_q + alpha * (0 + gamma * nolick_q - nolick_q)

#             lick_q, nolick_q = new_lick_q, new_nolick_q
            
#         elif action == 'NoLick':
#             reward = 0
            
#             # Record data
#             data.loc[i] = [lick_q, nolick_q, lick_prob, nolick_prob, action]
            
#             # Update Q-values using learning rate and discount factor
#             # new_lick_q = lick_q + alpha * (0 + gamma * (1*lick_prob )- lick_q)
#             new_lick_q = lick_q 
#             new_nolick_q = nolick_q + alpha * (0 + gamma * next_reward - nolick_q)

#             lick_q, nolick_q = new_lick_q, new_nolick_q
            
#     return data


def generate_action_reverse(init_lick_Q,init_Nolick_Q,action_len,Alpha,Gamma):

    '''
    更新迭代公式后的生成序列的计算过程

    init_lick_Q :lick 初始 Q 值
    init_Nolick_Q :nolick 初始 Q 值
    action_len: 模拟生成的行动序列长度
    '''
    # Set initial Q-values
    lick_q = init_lick_Q
    nolick_q = init_Nolick_Q
    # Iteration count
    N = action_len
    # Learning rate and discount factor
    alpha = Alpha
    THETA = 0.08
    THETA_p = 0.6
    PE = 0

    gamma = Gamma

    # Create an empty DataFrame to record data
    columns = ['Lick Q', 'NoLick Q', 'Lick Probability', 'NoLick Probability', 'Action','Alpha','PE']
    data = pd.DataFrame(columns=columns)

    # Iterate to generate data
    for i in range(N):
        # Convert Q-values to probabilities
        # lick_prob = qiyun_softmax_update([lick_q, nolick_q])[0]
        # nolick_prob = qiyun_softmax_update([lick_q, nolick_q])[1]
        # ============================================================
        Q = np.array([[lick_q, nolick_q]])
        probs = qiyun_softmax_update(Q)
        lick_prob = probs[0, 0]
        nolick_prob = probs[0, 1]
        # print(lick_prob,nolick_prob)
        # ============================================================
        # print(lick_prob)
        # Choose action based on probabilities
        action = np.random.choice(['Lick', 'NoLick'], p=[lick_prob, nolick_prob])

        # 当前动作的奖励
        if action == 'Lick':
            reward = -1
            
            # Record data
            data.loc[i] = [lick_q, nolick_q, lick_prob, nolick_prob, action, alpha, PE]
            
            # Update Q-values using learning rate and discount factor
            PE= reward - lick_q
            alpha = THETA_p*alpha+THETA*abs(PE)
            new_lick_q = lick_q + alpha * (reward - lick_q)
            # new_nolick_q = nolick_q + alpha * (0 + gamma * (1*nolick_prob)- nolick_q)
            new_nolick_q = nolick_q 

            # new_lick_q = lick_q + alpha * (reward + gamma * (lick_prob)) - lick_q)
            # new_nolick_q = nolick_q + alpha * (0 + gamma * nolick_q - nolick_q)

            lick_q, nolick_q = new_lick_q, new_nolick_q
            
        elif action == 'NoLick':
            reward = 0.1
            
            # Record data
            data.loc[i] = [lick_q, nolick_q, lick_prob, nolick_prob, action, alpha, PE]
            
            # Update Q-values using learning rate and discount factor
            # new_lick_q = lick_q + alpha * (0 + gamma * (1*lick_prob )- lick_q)

            PE= reward - nolick_q
            alpha = THETA_p*alpha+THETA*abs(PE)
            new_lick_q = lick_q 
            new_nolick_q = nolick_q + alpha * (reward + - nolick_q)

            lick_q, nolick_q = new_lick_q, new_nolick_q
            
    return data