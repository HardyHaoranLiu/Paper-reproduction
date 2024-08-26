# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:04:49 2024

@author: haorl
"""

from docplex.mp.model import Model

a = 10
homes = 2
c = [0.003, 0.006]
d = [0.42, 0.72]
D = [100, 200]

# 创建市场清算优化模型（上层问题）
def solve_market_clearing(bids):
    price = {}
    
    # 创建上层模型
    model = Model(name="market clearing")
    
    # 定义价格变量
    for home in range(homes):
        price[home] = model.continuous_var(name=f'price_{home}', lb=0)
    
    # 市场清算约束
    sharing_power_sum = model.sum(-a * price[home] + bids[home] for home in range(homes))
    model.add_constraint(sharing_power_sum == 0, 'market_clearing')
    
    # 定义目标函数：最小化所有家庭的价格成本之和
    market_cost = model.sum(price[home] * price[home] for home in range(homes))
    model.set_objective('min', market_cost)
    
    # 求解上层模型
    solution = model.solve()
    
    if solution:
        print("\nMarket clearing solution found:")
        prices = [solution[price[home]] for home in range(homes)]
        for home in range(homes):
            print(f'Home {home}: price_{home} = {prices[home]:.2f}')
        return prices
    else:
        print("No solution found for market clearing.")
        return None

# 主程序
def main():
    max_iterations = 100
    tolerance = 1e-5  # 收敛性检查阈值
    
    # 初始的假设出价（可以设置为零或其他合理的初值）
    bids = [0] * homes
    p = {}
    
    # 迭代求解双层优化问题
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}")

        # 求解上层问题（市场清算）
        prices = solve_market_clearing(bids)
        if prices is None:
            break
        
        # 更新下层问题的出价
        for home in range(homes):
            p[home] = (a * (homes - 1) * prices[home] - a * (homes - 1) * d[home] + D[home]) / (2 * a * (homes - 1) * c[home] + 1)
            bids[home] = D[home] - p[home] + a * prices[home]
        
        # 收敛性检查
        if all(abs(bids[home] - p[home]) < tolerance for home in range(homes)):
            print("Convergence achieved.")
            break
    
    print("Final bids:", bids)

if __name__ == "__main__":
    main()
