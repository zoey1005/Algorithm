from collections import deque
import pandas
import requests
import time


class GraphSearcher:
    def __init__(self):
        self.visited = set()
        self.order = []

    def visit_and_get_children(self, node):
        """ Record the node value in self.order, and return its children
        param: node
        return: children of the given node
        """
        raise Exception("must be overridden in sub classes -- don't change me here!")

    def dfs_search(self, node):
        self.visited.clear()
        self.order.clear()
        return self.dfs_visit(node)  # 2. start recursive search by calling dfs_visit

    def dfs_visit(self, node):
        if node in self.visited:
            return None
        else:
            self.visited.add(node)
            children = self.visit_and_get_children(node)
            for w in children:
                self.dfs_visit(w)

    def bfs_search(self, node):
        todo_queue = deque([node])
        added = {node}
        while len(todo_queue) > 0:
            curr_node = todo_queue.popleft()
            children = self.visit_and_get_children(curr_node)
            for w in children:
                if not (w in added):
                    todo_queue.append(w)
                    added.add(w)


class MatrixSearcher(GraphSearcher):
    def __init__(self, df):
        super().__init__()  # call constructor method of parent class
        self.df = df

    def visit_and_get_children(self, node):
        # for node not in self.order:
        self.order.append(node)
        # TODO: Record the node value in self.order
        children = []
        for node, has_edge in self.df.loc[node].items():
            if has_edge == 1:
                children.append(node)
        return children


class FileSearcher(GraphSearcher):
    def __init__(self):
        super().__init__()

    def visit_and_get_children(self, node):
        with open('file_nodes/' + node, 'r', encoding='utf-8') as f:
            content = f.read().splitlines()
            f.close()
            self.order.append(content[0])
            children = content[1].split(',')
        return children

    def concat_order(self):
        if len(self.order) == 0:
            return ''
        return ''.join(self.order)


class WebSearcher(GraphSearcher):
    def __init__(self, driver):
        super().__init__()
        self.driver = driver
        self.pandases = []

    def visit_and_get_children(self, node):
        # self.order.append(node)
        self.driver.get(node)
        urls = self.driver.find_elements_by_xpath("//a")
        url_list = []
        for url in urls:
            link = url.get_attribute("href")

            url_list.append(link)
        self.order.append(node)

        self.pandases.append(pandas.read_html(node)[0])

        return url_list

    def table(self):
        return pandas.concat(self.pandases, ignore_index=True)


def reveal_secrets(driver, url, travellog):
    # s = WebSearcher(driver)
    # s.bfs_search(url)
    series = travellog['clue']
    password_ = series.tolist()
    password = ''
    for x in password_:
        password += str(x)
    driver.get(url)
    text = driver.find_element("id", "password")
    btn = driver.find_element("id", "attempt-button")
    text.send_keys(password)
    btn.click()
    time.sleep(2)
    btn2 = driver.find_element("id", "securityBtn")
    btn2.click()
    time.sleep(2)
    urls_ = driver.find_elements_by_tag_name("img")[0].get_attribute("src")
    r = requests.get(str(urls_))
    with open("Current_Location.jpg", "wb") as f:
        f.write(r.content)
    image_ = driver.find_element("id", "location").text
    return image_

# image=driver.find_element("id", "image")
