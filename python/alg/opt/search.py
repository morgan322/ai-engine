#https://www.cnblogs.com/lsqin/p/9342929.html

# 最基础的遍历无序列表的查找算法
# 时间复杂度O(n)
 
def sequential_search(lis, key):
  length = len(lis)
  for i in range(length):
    if lis[i] == key:
      return i
    else:
      return False
    
# 针对有序查找表的二分查找算法
def binary_search(lis, key):
    low = 0
    high = len(lis) - 1
    time = 0
    while low < high:
        time += 1
    mid = int((low + high) / 2)
    if key < lis[mid]:
        high = mid - 1
    elif key > lis[mid]:
        low = mid + 1
    else:
        # 打印折半的次数
        print("times: %s" % time)
        return mid
    print("times: %s" % time)
    return False
# 插值查找算法
 
def binary_search(lis, key):
  low = 0
  high = len(lis) - 1
  time = 0
  while low < high:
    time += 1
    # 计算mid值是插值算法的核心代码
    mid = low + int((high - low) * (key - lis[low])/(lis[high] - lis[low]))
    print("mid=%s, low=%s, high=%s" % (mid, low, high))
    if key < lis[mid]:
      high = mid - 1
    elif key > lis[mid]:
      low = mid + 1
    else:
      # 打印查找的次数
      print("times: %s" % time)
      return mid
  print("times: %s" % time)
  return False
 
# 斐波那契查找算法
# 时间复杂度O(log(n))
def fibonacci_search(lis, key):
    # 需要一个现成的斐波那契列表。其最大元素的值必须超过查找表中元素个数的数值。
    F = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,
    233, 377, 610, 987, 1597, 2584, 4181, 6765,
    10946, 17711, 28657, 46368]
    low = 0
    high = len(lis) - 1
    
    # 为了使得查找表满足斐波那契特性，在表的最后添加几个同样的值
    # 这个值是原查找表的最后那个元素的值
    # 添加的个数由F[k]-1-high决定
    k = 0
    while high > F[k]-1:
        k += 1
    print(k)
    i = high
    while F[k]-1 > i:
        lis.append(lis[high])
        i += 1
    print(lis)
    
    # 算法主逻辑。time用于展示循环的次数。
    time = 0
    while low <= high:
        time += 1
        # 为了防止F列表下标溢出，设置if和else
    if k < 2:
        mid = low
    else:
        mid = low + F[k-1]-1
    
    print("low=%s, mid=%s, high=%s" % (low, mid, high))
    if key < lis[mid]:
        high = mid - 1
        k -= 1
    elif key > lis[mid]:
        low = mid + 1
        k -= 2
    else:
        if mid <= high:
            # 打印查找的次数
            print("times: %s" % time)
            return mid
        else:
            print("times: %s" % time)
            return high
    print("times: %s" % time)
    return False

# 二叉树查找 Python实现
class BSTNode:
    """
    定义一个二叉树节点类。
    以讨论算法为主，忽略了一些诸如对数据类型进行判断的问题。
    """
    def __init__(self, data, left=None, right=None):
        """
        初始化
        :param data: 节点储存的数据
        :param left: 节点左子树
        :param right: 节点右子树
        """
        self.data = data
        self.left = left
        self.right = right


class BinarySortTree:
    """
    基于BSTNode类的二叉查找树。维护一个根节点的指针。
    """
    def __init__(self):
        self._root = None

    def is_empty(self):
        return self._root is None

    def search(self, key):
        """
        关键码检索
        :param key: 关键码
        :return: 查询节点或None
        """
        bt = self._root
        while bt:
            entry = bt.data
            if key < entry:
                bt = bt.left
            elif key > entry:
                bt = bt.right
            else:
                return entry
        return None

    def insert(self, key):
        """
        插入操作
        :param key:关键码 
        :return: 布尔值
        """
        bt = self._root
        if not bt:
            self._root = BSTNode(key)
            return
        while True:
            entry = bt.data
            if key < entry:
                if bt.left is None:
                    bt.left = BSTNode(key)
                    return
                bt = bt.left
            elif key > entry:
                if bt.right is None:
                    bt.right = BSTNode(key)
                    return
                bt = bt.right
            else:
                bt.data = key
                return

    def delete(self, key):
        """
        二叉查找树最复杂的方法
        :param key: 关键码
        :return: 布尔值
        """
        p, q = None, self._root     # 维持p为q的父节点，用于后面的链接操作
        if not q:
            print("空树！")
            return
        while q and q.data != key:
            p = q
            if key < q.data:
                q = q.left
            else:
                q = q.right
            if not q:               # 当树中没有关键码key时，结束退出。
                return
        # 上面已将找到了要删除的节点，用q引用。而p则是q的父节点或者None（q为根节点时）。
        if not q.left:
            if p is None:
                self._root = q.right
            elif q is p.left:
                p.left = q.right
            else:
                p.right = q.right
            return
        # 查找节点q的左子树的最右节点，将q的右子树链接为该节点的右子树
        # 该方法可能会增大树的深度，效率并不算高。可以设计其它的方法。
        r = q.left
        while r.right:
            r = r.right
        r.right = q.right
        if p is None:
            self._root = q.left
        elif p.left is q:
            p.left = q.left
        else:
            p.right = q.left

    def __iter__(self):
        """
        实现二叉树的中序遍历算法,
        展示我们创建的二叉查找树.
        直接使用python内置的列表作为一个栈。
        :return: data
        """
        stack = []
        node = self._root
        while node or stack:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            yield node.data
            node = node.right

class Node(object):
    def __init__(self,key):
        self.key1=key
        self.key2=None
        self.left=None
        self.middle=None
        self.right=None
    def isLeaf(self):
        return self.left is None and self.middle is None and self.right is None
    def isFull(self):
        return self.key2 is not None
    def hasKey(self,key):
        if (self.key1==key) or (self.key2 is not None and self.key2==key):
            return True
        else:
            return False
    def getChild(self,key):
        if key<self.key1:
            return self.left
        elif self.key2 is None:
            return self.middle
        elif key<self.key2:
            return self.middle
        else:
            return self.right

class two_three_Tree(object):
    def __init__(self):
        self.root=None
    def get(self,key):
        if self.root is None:
            return None
        else:
            return self._get(self.root,key)
    def _get(self,node,key):
        if node is None:
            return None
        elif node.hasKey(key):
            return node
        else:
            child=node.getChild(key)
            return self._get(child,key)
    def put(self,key):
        if self.root is None:
            self.root=Node(key)
        else:
            pKey,pRef=self._put(self.root,key)
            if pKey is not None:
                newnode=Node(pKey)
                newnode.left=self.root
                newnode.middle=pRef
                self.root=newnode
    def _put(self,node,key):
        if node.hasKey(key):
            return None,None
        elif node.isLeaf():
            return self._addtoNode(node,key,None)
        else:
            child=node.getChild(key)
            pKey,pRef=self._put(child,key)
            if pKey is None:
                return None,None
            else:
                return self._addtoNode(node,pKey,pRef)
             
         
    def _addtoNode(self,node,key,pRef):
        if node.isFull():
            return self._splitNode(node,key,pRef)
        else:
            if key<node.key1:
                node.key2=node.key1
                node.key1=key
                if pRef is not None:
                    node.right=node.middle
                    node.middle=pRef
            else:
                node.key2=key
                if pRef is not None:
                    node.right=Pref
            return None,None
    def _splitNode(self,node,key,pRef):
        newnode=Node(None)
        if key<node.key1:
            pKey=node.key1
            node.key1=key
            newnode.key1=node.key2
            if pRef is not None:
                newnode.left=node.middle
                newnode.middle=node.right
                node.middle=pRef
        elif key<node.key2:
            pKey=key
            newnode.key1=node.key2
            if pRef is not None:
                newnode.left=Pref
                newnode.middle=node.right
        else:
            pKey=node.key2
            newnode.key1=key
            if pRef is not None:
                newnode.left=node.right
                newnode.middle=pRef
        node.key2=None
        return pKey,newnode
#红黑树
from random import randint

RED = 'red'
BLACK = 'black'

class RBT:
    def __init__(self):
       # self.items = []
        self.root = None
        self.zlist = []

    def LEFT_ROTATE(self, x):
        # x是一个RBTnode
        y = x.right
        if y is None:
            # 右节点为空，不旋转
            return
        else:
            beta = y.left
            x.right = beta
            if beta is not None:
                beta.parent = x

            p = x.parent
            y.parent = p
            if p is None:
                # x原来是root
                self.root = y
            elif x == p.left:
                p.left = y
            else:
                p.right = y
            y.left = x
            x.parent = y

    def RIGHT_ROTATE(self, y):
        # y是一个节点
        x = y.left
        if x is None:
            # 右节点为空，不旋转
            return
        else:
            beta = x.right
            y.left = beta
            if beta is not None:
                beta.parent = y

            p = y.parent
            x.parent = p
            if p is None:
                # y原来是root
                self.root = x
            elif y == p.left:
                p.left = x
            else:
                p.right = x
            x.right = y
            y.parent = x

    def INSERT(self, val):

        z = RBTnode(val)
        y = None
        x = self.root
        while x is not None:
            y = x
            if z.val < x.val:
                x = x.left
            else:
                x = x.right

        z.PAINT(RED)
        z.parent = y

        if y is None:
            # 插入z之前为空的RBT
            self.root = z
            self.INSERT_FIXUP(z)
            return

        if z.val < y.val:
            y.left = z
        else:
            y.right = z

        if y.color == RED:
            # z的父节点y为红色，需要fixup。
            # 如果z的父节点y为黑色，则不用调整
            self.INSERT_FIXUP(z)

        else:
            return

    def INSERT_FIXUP(self, z):
        # case 1:z为root节点
        if z.parent is None:
            z.PAINT(BLACK)
            self.root = z
            return

        # case 2:z的父节点为黑色
        if z.parent.color == BLACK:
            # 包括了z处于第二层的情况
            # 这里感觉不必要啊。。似乎z.parent为黑色则不会进入fixup阶段
            return

        # 下面的几种情况，都是z.parent.color == RED:
        # 节点y为z的uncle
        p = z.parent
        g = p.parent  # g为x的grandpa
        if g is None:
            return
            #   return 这里不能return的。。。
        if g.right == p:
            y = g.left
        else:
            y = g.right

        # case 3-0:z没有叔叔。即：y为NIL节点
        # 注意，此时z的父节点一定是RED
        if y == None:
            if z == p.right and p == p.parent.left:
                # 3-0-0:z为右儿子,且p为左儿子，则把p左旋
                # 转化为3-0-1或3-0-2的情况
                self.LEFT_ROTATE(p)
                p, z = z, p
                g = p.parent
            elif z == p.left and p == p.parent.right:
                self.RIGHT_ROTATE(p)
                p, z = z, p

            g.PAINT(RED)
            p.PAINT(BLACK)
            if p == g.left:
                # 3-0-1:p为g的左儿子
                self.RIGHT_ROTATE(g)
            else:
                # 3-0-2:p为g的右儿子
                self.LEFT_ROTATE(g)

            return

        # case 3-1:z有黑叔
        elif y.color == BLACK:
            if p.right == z and p.parent.left == p:
                # 3-1-0:z为右儿子,且p为左儿子,则左旋p
                # 转化为3-1-1或3-1-2
                self.LEFT_ROTATE(p)
                p, z = z, p
            elif p.left == z and p.parent.right == p:
                self.RIGHT_ROTATE(p)
                p, z = z, p

            p = z.parent
            g = p.parent

            p.PAINT(BLACK)
            g.PAINT(RED)
            if p == g.left:
                # 3-1-1:p为g的左儿子，则右旋g
                self.RIGHT_ROTATE(g)
            else:
                # 3-1-2:p为g的右儿子，则左旋g
                self.LEFT_ROTATE(g)

            return


        # case 3-2:z有红叔
        # 则涂黑父和叔，涂红爷，g作为新的z，递归调用
        else:
            y.PAINT(BLACK)
            p.PAINT(BLACK)
            g.PAINT(RED)
            new_z = g
            self.INSERT_FIXUP(new_z)

    def DELETE(self, val):
        curNode = self.root
        while curNode is not None:
            if val < curNode.val:
                curNode = curNode.left
            elif val > curNode.val:
                curNode = curNode.right
            else:
                # 找到了值为val的元素,正式开始删除

                if curNode.left is None and curNode.right is None:
                    # case1:curNode为叶子节点：直接删除即可
                    if curNode == self.root:
                        self.root = None
                    else:
                        p = curNode.parent
                        if curNode == p.left:
                            p.left = None
                        else:
                            p.right = None

                elif curNode.left is not None and curNode.right is not None:
                    sucNode = self.SUCCESOR(curNode)
                    curNode.val, sucNode.val  = sucNode.val, curNode.val
                    self.DELETE(sucNode.val)

                else:
                    p = curNode.parent
                    if curNode.left is None:
                        x = curNode.right
                    else:
                        x = curNode.left
                    if curNode == p.left:
                        p.left = x
                    else:
                        p.right = x
                    x.parent = p
                    if curNode.color == BLACK:
                        self.DELETE_FIXUP(x)

                curNode = None
        return False

    def DELETE_FIXUP(self, x):
        p = x.parent
        # w:x的兄弟结点
        if x == p.left:
            w = x.right
        else:
            w = x.left

        # case1:x的兄弟w是红色的
        if w.color == RED:
            p.PAINT(RED)
            w.PAINT(BLACK)
            if w == p.right:
                self.LEFT_ROTATE(p)
            else:
                self.RIGHT_ROTATE(p)

        if w.color == BLACK:
            # case2:x的兄弟w是黑色的，而且w的两个孩子都是黑色的
            if w.left.color == BLACK and w.right.color == BLACK:
                w.PAINT(RED)
                if p.color == BLACK:
                    return
                else:
                    p.color = BLACK
                    self.DELETE_FIXUP(p)

            # case3:x的兄弟w是黑色的，而且w的左儿子是红色的，右儿子是黑色的
            if w.left.color == RED and w.color == BLACK:
                w.left.PAINT(BLACK)
                w.PAINT(RED)
                self.RIGHT_ROTATE(w)

            # case4:x的兄弟w是黑色的，而且w的右儿子是红
            if w.right.color == RED:
                p.PAINT(BLACK)
                w.PAINT(RED)
                if w == p.right:
                    self.LEFT_ROTATE(p)
                else:
                    self.RIGHT_ROTATE(p)

    def SHOW(self):
        self.DISPLAY1(self.root)
        return self.zlist

    def DISPLAY1(self, node):
        if node is None:
            return
        self.DISPLAY1(node.left)
        self.zlist.append(node.val)
        self.DISPLAY1(node.right)

    def DISPLAY2(self, node):
        if node is None:
            return
        self.DISPLAY2(node.left)
        print(node.val)
        self.DISPLAY2(node.right)

    def DISPLAY3(self, node):
        if node is None:
            return
        self.DISPLAY3(node.left)
        self.DISPLAY3(node.right)
        print(node.val)

class RBTnode:
    '''红黑树的节点类型'''
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None

    def PAINT(self, color):
        self.color = color

def zuoxuan(b, c):
    a = b.parent
    a.left = c
    c.parent = a
    b.parent = c
    c.left = b
# -*- coding: UTF-8 -*-
# B树查找

class BTree:  #B树
    def __init__(self,value):
        self.left=None
        self.data=value
        self.right=None

    def insertLeft(self,value):
        self.left=BTree(value)
        return self.left

    def insertRight(self,value):
        self.right=BTree(value)
        return self.right

    def show(self):
        print(self.data)


def inorder(node):  #中序遍历：先左子树，再根节点，再右子树
    if node.data:
        if node.left:
            inorder(node.left)
        node.show()
        if node.right:
            inorder(node.right)


def rinorder(node):  #倒中序遍历
    if node.data:
        if node.right:
            rinorder(node.right)
        node.show()
        if node.left:
            rinorder(node.left)

def insert(node,value):
    if value > node.data:
        if node.right:
            insert(node.right,value)
        else:
            node.insertRight(value)
    else:
        if node.left:
            insert(node.left,value)
        else:
            node.insertLeft(value)
# 忽略了对数据类型，元素溢出等问题的判断。

class HashTable:
    def __init__(self, size):
        self.elem = [None for i in range(size)] # 使用list数据结构作为哈希表元素保存方法
        self.count = size # 最大表长
    
    def hash(self, key):
        return key % self.count # 散列函数采用除留余数法
    
    def insert_hash(self, key):
        """插入关键字到哈希表内"""
        address = self.hash(key) # 求散列地址
        while self.elem[address]: # 当前位置已经有数据了，发生冲突。
            address = (address+1) % self.count # 线性探测下一地址是否可用
        self.elem[address] = key # 没有冲突则直接保存。
    
    def search_hash(self, key):
        """查找关键字，返回布尔值"""
        star = address = self.hash(key)
        while self.elem[address] != key:
            address = (address + 1) % self.count
        if not self.elem[address] or address == star: # 说明没找到或者循环到了开始的位置
            return False
        return True

if __name__ == '__main__':
    list_a = [12, 67, 56, 16, 25, 37, 22, 29, 15, 47, 48, 34]
    hash_table = HashTable(12)
    for i in list_a:
        hash_table.insert_hash(i)
    
    for i in hash_table.elem:
        if i:
            print((i, hash_table.elem.index(i)), end=" ")
    print("\n")
    
    print(hash_table.search_hash(15))
    # print(hash_table.search_hash(33))



    l=[88,11,2,33,22,4,55,33,221,34]
    Root=BTree(l[0])
    node=Root
    for i in range(1,len(l)):
        insert(Root,l[i])

    print("中序遍历（从小到大排序 ）")
    inorder(Root)
    print("倒中序遍历（从大到小排序）")
    rinorder(Root)

    rbt=RBT()
    b = []

    for i in range(100):
        m = randint(0, 500)
        rbt.INSERT(m)
        b.append(m)

    a = rbt.SHOW()
    b.sort()
    equal = True
    for i in range(100):
        if a[i] != b[i]:
            equal = False
            break

    if not equal:
        print('wrong')
    else:
        print('OK!')


    LIST = [62, 58, 88, 48, 73, 99, 35, 51, 93, 29, 37, 49, 56, 36, 50]
    bs_tree = BinarySortTree()
    for i in range(len(LIST)):
        bs_tree.insert(LIST[i])
    bs_tree.insert(100)
    bs_tree.delete(58)
    for i in bs_tree:
        print(i, end=" ")
    print("\n", bs_tree.search(4))
    # result = fibonacci_search(LIST, 444)
    # print(result)
    # result = binary_search(LIST, 444)
    # print(result)
    # # result = binary_search(LIST, 99)
    # # print(result)
    # result = sequential_search(LIST, 123)
    # print(result)
