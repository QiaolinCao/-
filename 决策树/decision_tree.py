"""
@author: Qiaolin Cao
@time: 2024.03.08
参考周志华《机器学习》第四章
"""
from typing import List, Dict, Optional, Deque, Callable, Union
import pandas as pd
import numpy as np
from collections import deque
from dataclasses import dataclass
import copy


def information_entropy(y: pd.Series) -> float:
    ratios: pd.Series = y.value_counts() / len(y)
    ent = -ratios.apply(lambda ratio: ratio * np.log2(ratio)).sum()
    return ent


def gini_value(ys: pd.Series) -> float:
    ratios: pd.Series = ys.value_counts() / len(ys)
    gini = 1 - np.power(ratios, 2).sum()
    return gini


@dataclass
class DataStructure:
    continuous_attributes: List[str]
    discrete_attributes: List[str]
    category: str


class EquationSet:
    """
    用于描述连续变量的取值范围，默认左开右闭，如 3 < x <= 5
    """
    def __init__(self, sign: str, value: float) -> None:
        self.equations: Dict[str, float] = {}
        self.acceptable = {"<=": "lt", "lt": "lt", ">": "gt", "gt": "gt"}
        if sign not in self.acceptable.keys():
            raise ValueError("sign参数错误")
        sign = self.acceptable[sign]
        self.equations[sign] = value

    def append_equation(self, sign: str, value: float) -> None:
        """
        添加新的等式
        """
        if sign not in self.acceptable.keys():
            raise ValueError("sign参数错误")
        sign = self.acceptable[sign]

        if sign not in self.equations.keys():
            self.equations[sign] = value
        else:
            if sign == "lt":
                # 确保不矛盾
                gt_value = self.equations.get("gt")
                if gt_value is not None:
                    if value <= gt_value:
                        return
                if value < self.equations[sign]:
                    self.equations[sign] = value
            else:
                lt_value = self.equations.get("lt")
                if lt_value is not None:
                    if lt_value <= value:
                        return
                if value > self.equations[sign]:
                    self.equations[sign] = value

    def get_equations(self) -> Dict[str, float]:
        return self.equations


class DecisionTree:
    """
    决策树
    """
    def __init__(
            self,
            divide_method: str,
            pruning_method: str,
            search_method: str,
            max_depth: int,
            max_node: int
    ) -> None:
        self._node_id: int = 0
        self.nodes: Dict[str, Node] = {}
        self._temp_termination: Deque[Node] = deque()
        self.leaf_nodes: Dict[str, Node] = {}

        self.divide_method: str = divide_method
        self.pruning_method: str = pruning_method
        self.search_method: str = search_method
        self.max_depth: int = max_depth
        self.max_node: int = max_node

        self.train_set: Optional[pd.DataFrame] = None
        self.test_set: Optional[pd.DataFrame] = None
        self.data_structure: Optional[DataStructure] = None

    def gen_new_node_id(self) -> int:
        self._node_id += 1
        return self._node_id - 1

    def add_node(self, node: "Node") -> None:
        self.nodes[str(node.id)] = node

    def add_leaf_node(self, node: "Node") -> None:
        self.leaf_nodes[str(node.id)] = node

    def add_temp_termination(self, node: "Node") -> None:
        """
        向队列尾端添加临时终结点
        """
        self._temp_termination.append(node)

    def del_node(self, node: "Node") -> None:
        node_id: str = str(node.id)
        del self.nodes[node_id]
        if node_id in self.leaf_nodes:
            del self.leaf_nodes[node_id]
        if node in self._temp_termination:
            self._temp_termination.remove(node)

        self._node_id -= 1

    def get_test_set_accuracy(self) -> float:
        # 判断节点为所有叶节点，与临时终结点
        df: pd.DataFrame = self.test_set
        for container in [self.leaf_nodes.values(), self._temp_termination]:
            for node in container:
                # 筛选出符合判断节点的测试集数据，并赋予prediction
                conditions: pd.Series = pd.Series(True, index=df.index)
                for col, fixed_value in node.fixed_attributes.items():
                    # 连续变量
                    if isinstance(fixed_value, EquationSet):
                        for sign, value in fixed_value.get_equations().items():
                            if sign == "lt":
                                conditions &= (df[col] <= value)
                            else:
                                # greater than
                                conditions %= (df[col] > value)
                    # 离散变量
                    else:
                        conditions &= (df[col] == fixed_value)
                df.loc[conditions, "prediction"] = node.node_prediction
        accuracy: float = (df["prediction"] == df[self.data_structure.category]).sum() / len(df)
        return accuracy

    def fit(
            self,
            train_set: pd.DataFrame,
            test_set: pd.DataFrame,
            data_structure: DataStructure
    ) -> None:
        """
        train_set传入数据最后一列为category，其他列需为属性
        test_set所有列均为属性
        """
        # 预检查
        for col_name in train_set.columns:
            if "ct." in col_name:
                raise ValueError("'ct.'用于内置命名，列名请勿包含'ct.'")

        self.train_set = train_set
        self.test_set = test_set
        self.data_structure = data_structure
        root_node: Node = Node(
            self,
            list(train_set.columns[:-1]),
            {},
            self.train_set,
        )
        self.add_node(root_node)

        if self.search_method == "depth_first":
            # 降序队列右侧添加，右侧拿取，后进先出，实现深度优先
            get_node: Callable = self._temp_termination.pop
        elif self.search_method == "breadth_first":
            # 队列右侧添加，左侧拿取，先进先出，实现广度优先
            get_node: Callable = self._temp_termination.popleft
        else:
            raise ValueError(f"不支持搜索方式{self.search_method}")

        while self._temp_termination:
            if self.pruning_method == "prepruning":
                current_accuracy: float = self.get_test_set_accuracy()
                node: Node = get_node()
                child_lst: List[Node] = node.create_child_nodes()
                accuracy_after_division: float = self.get_test_set_accuracy()
                if accuracy_after_division >= current_accuracy:
                    # 不剪枝
                    continue
                else:
                    # 剪枝，并将节点改为叶节点
                    while child_lst:
                        child_node: Node = child_lst.pop()
                        child_node.parent.del_child(child_node)
                        self.del_node(child_node)
                    self.add_leaf_node(node)

            else:
                node: Node = get_node()
                node.create_child_nodes()


class Node:
    """
    决策树的每个节点
    """
    def __init__(
            self,
            tree: DecisionTree,
            free_attrs: List[str],
            fixed_attrs: Dict[str, Union[int, EquationSet]],
            data: pd.DataFrame,
            parent: Optional["Node"] = None,
            depth: int = 0
    ) -> None:
        self.tree: DecisionTree = tree
        self.id: int = self.tree.gen_new_node_id()
        self.tree.add_node(self)

        self.free_attributes: List[str] = free_attrs
        self.fixed_attributes: Dict[str, Union[int, EquationSet]] = fixed_attrs
        self.data: pd.DataFrame = data
        self.depth: int = depth

        self.parent: Optional[Node] = parent
        self.children: Dict[str, Node] = {}

        self.is_leaf: bool = self.check_is_leaf()
        self.node_prediction: int = self._get_prediction()
        if self.is_leaf:
            self.tree.add_leaf_node(self)
        else:
            self.tree.add_temp_termination(self)

    def check_is_leaf(self) -> bool:
        # 若该划分下，样本为空
        category_col: str = self.tree.data_structure.category
        if len(self.data) == 0:
            return True
        # 若该划分下，所有样本均属同一类别
        elif len(self.data[category_col].unique()) == 1:
            return True
        # 若没有可供进一步划分的free_attributes
        elif not self.free_attributes:
            return True
        # 若该划分下，所有样本属性取值均相同
        elif len(self.data[self.free_attributes].drop_duplicates()) == 1:
            return True
        else:
            return False

    def add_child(self, node: "Node") -> None:
        node_id: str = str(node.id)
        self.children[node_id] = node

    def del_child(self, node: "Node") -> None:
        node_id: str = str(node.id)
        del self.children[node_id]

    def information_gain(self, attr: str) -> float:
        """
        计算信息增益
        :param attr: 划分所依据的属性；必须为free_attributes中的属性, 且为离散属性
        :return:
        """
        groups = self.data.groupby(attr)
        category_col: str = self.tree.data_structure.category

        ent = information_entropy(self.data[category_col])

        attr_ratios: pd.Series = groups.apply(len) / len(self.data)
        sub_ent: pd.Series = groups[category_col].agg(information_entropy)

        information_gain = ent - (attr_ratios * sub_ent).sum()
        return information_gain

    def intrinsic_value(self, attr: str) -> float:
        """
        增益率计算过程中的intrinsic value；传入属性需为离散变量
        """
        groups = self.data.groupby(attr)
        attr_ratios: pd.Series = groups.apply(len) / len(self.data)
        intrinsic_value = - (attr_ratios * np.log2(attr_ratios)).sum()
        return intrinsic_value

    def gain_ratio(self, attr: str) -> float:
        """
        计算增益率; 传入属性需为离散变量
        """
        return self.information_gain(attr) / self.intrinsic_value(attr)

    def gini_index(self, attr: str) -> float:
        """
        计算基尼指数；传入变量需为离散变量
        """
        groups = self.data.groupby(attr)
        category_col: str = self.tree.data_structure.category

        attr_ratios: pd.Series = groups.apply(len) / len(self.data)
        gini_index = (attr_ratios * groups[category_col].agg(gini_value)).sum()
        return gini_index

    def _get_prediction(self) -> int:
        """
        如果是叶节点，返回该节点的预测类型值
        """

        category_series: pd.Series = self.data[self.tree.data_structure.category]
        if len(category_series) == 0:
            category_series = self.parent.data[self.tree.data_structure.category]
        prediction: int = category_series.mode()[0]
        return prediction

    def _get_optimal_division(self) -> str:
        """
        :return: 最优划分属性
        """
        attributes: list = []
        method: str = self.tree.divide_method

        # 仅在free_attributes中寻找最优划分属性
        data_df: pd.DataFrame = self.data.copy()
        data_df = data_df[self.free_attributes + [self.tree.data_structure.category]]

        # 将连续变量离散化；目前仅有bi_partition的的离散化方式
        def _get_cut_points(ser: pd.Series) -> List[float]:
            ser = ser.drop_duplicates().sort_values().reset_index(drop=True)
            return list(((ser + ser.shift(1)) / 2).dropna())

        attrs_to_discrete: List[str] = [attr for attr in self.tree.data_structure.continuous_attributes
                                        if attr in data_df.columns]
        for attr in attrs_to_discrete:
            cut_points: List[float] = _get_cut_points(data_df[attr])
            for point in cut_points:
                col_name = f"ct.{attr}_lt_{point}"
                # 以ct.开头，方便后续区分，该属性是否是连续变量
                data_df[col_name] = (data_df[attr] <= point).astype(int)
        data_df.drop(columns=attrs_to_discrete, inplace=True)

        # 依据divide_method，寻找最优划分属性
        if method == "gini_index":
            gini_indexes: list = []
            for attr in self.free_attributes:
                attributes.append(attr)
                gini_indexes.append(self.gini_index(attr))
            df = pd.DataFrame(index=attributes, data={"gini_index": gini_indexes})
            optimal_division: str = df["gini_index"].idxmin()

        elif method == "information_gain":
            gains: list = []
            for attr in self.free_attributes:
                attributes.append(attr)
                gains.append(self.information_gain(attr))
            df = pd.DataFrame(index=attributes, data={"information_gain": gains})
            optimal_division: str = df["information_gain"].idxmax()

        elif method == "gain_ratio":
            gains: list = []
            for attr in self.free_attributes:
                attributes.append(attr)
                gains.append(self.information_gain(attr))
            df = pd.DataFrame(index=attributes, data={"information_gain": gains})

            # 启发式算法：先找出信息增益高于平均水平的属性，在从中选择增益率最高的
            mean: float = df["information_gain"].mean()
            df = df[df["information_gain"] >= mean]
            for attr in df.index:
                df.loc[attr, "IV"] = self.intrinsic_value(attr)

            df["gain_ratio"] = df["information_gain"] / df["IV"]
            optimal_division: str = df["gain_ratio"].idxmax()

        else:
            raise ValueError(f"不支持划分方法：{method}")

        return optimal_division

    def create_child_nodes(self) -> List["Node"]:
        child_lst: List[Node] = []
        # 限制情况
        if self.depth == self.tree.max_depth:
            return child_lst
        node_nums: int = len(self.tree.nodes)
        if node_nums > self.tree.max_node:
            return child_lst

        optimal_division: str = self._get_optimal_division()

        # 判断创建子结点后，加入_temp_termination的方式是升序还是降序
        if self.tree.search_method == "depth_first":
            is_reverse = True
        else:
            is_reverse = False

        # 判断待划分属性是离散变量还是连续变量
        # 连续变量
        if optimal_division.startswith("ct."):
            optimal_division = optimal_division.replace("ct.", "")
            div_attr, _, value = optimal_division.split("_")
            free_attrs: List[str] = copy.copy(self.free_attributes)

            # 检测进行本此划分后，连续变量是否还是自由变量
            # 如果划分完后，不再是自由变量，则从free_attributes 中删除
            if len(self.data[div_attr].unique()) == 2:
                free_attrs.remove(div_attr)

            # 依据连续变量 <= 分割点，或大于分割点， 划分子节点
            signs = ["gt", "lt"]
            sign1, sign2 = signs if is_reverse else signs[::-1]
            fixed_attrs_1: Dict[str, Union[int, EquationSet]] = copy.copy(self.fixed_attributes)
            fixed_attrs_2: Dict[str, Union[int, EquationSet]] = copy.copy(self.fixed_attributes)
            if div_attr not in self.fixed_attributes.keys():
                fixed_attrs_1[div_attr] = EquationSet(sign1, value)
                fixed_attrs_2[div_attr] = EquationSet(sign2, value)
            else:
                fixed_attrs_1[div_attr].append_equation(sign1, value)
                fixed_attrs_1[div_attr].append_equation(sign2, value)

            data_lst = [self.data[self.data[div_attr] > value], self.data[self.data[div_attr] <= value]]
            data1, data2 = data_lst if is_reverse else data_lst[::-1]

            node_1 = Node(
                self.tree,
                free_attrs,
                fixed_attrs_1,
                data1,
                self,
                self.depth + 1
            )

            node_2 = Node(
                self.tree,
                free_attrs,
                fixed_attrs_2,
                data2,
                self,
                self.depth + 1
            )

            self.add_child(node_1)
            self.add_child(node_2)

        # 离散变量
        else:
            value_range: List[int] = sorted(self.data[optimal_division].unique(), reverse=is_reverse)
            free_attrs: List[str] = copy.copy(self.free_attributes)
            free_attrs.remove(optimal_division)
            for value in value_range:
                fixed_attrs: Dict[str, Union[int, EquationSet]] = copy.copy(self.fixed_attributes)
                fixed_attrs[optimal_division] = value
                data = self.data[self.data[optimal_division] == value]
                node = Node(
                    self.tree,
                    free_attrs,
                    fixed_attrs,
                    data,
                    self,
                    self.depth + 1
                )
                self.add_child(node)

        return list(self.children.values())

