import sys
from os import path


class Tree(object):

    def __init__(self, root):
        self.root = root

    @staticmethod
    def load_from_taems(filename):

        with open(filename, 'r') as file:
            type_node = None
            root = None
            values = {}
            nodes = {}
            multiline = False

            for line in file:
                line = line.strip()

                if not line or line.startswith('%'):
                    continue

                if type_node is None:
                    type_node = line[6:].strip()
                    if type_node != 'method' and type_node != 'task_group':
                        break
                else:
                    if line.startswith('(') and line.endswith(')'):
                        if not multiline:
                            parts = line[1:-1].split(' ', 1)

                            if len(parts) != 2:
                                raise ValueError("Illegal input format for single line")
                                sys.exit(1)

                            values[parts[0]] = parts[1]
                    elif line.startswith('('):
                        multiline = True
                    elif line.endswith(')'):
                        if multiline:
                            multiline = False
                            continue

                        children = values.get('subtasks', '').split(', ')
                        label = values['label']
                        nodes[label] = Node(type_node, label, values['agent'], values.get('qaf'), children)

                        if 'supertasks' not in values:
                            root = nodes[label]

                        type_node = None
                        values = {}

            for node in nodes.values():
                node.children = [nodes[child] for child in node.children if child in nodes]

            return Tree(root)

    def damp_to_dot(self, filename):
        with open(filename, 'w') as file:
            file.write('digraph BST {\n')

            self._damp_to_dot(file, self.root)

            file.write('}\n')


    def _damp_to_dot(self, file, node):
        file.write(node.to_dot())

        for child in node.children:
            self._damp_to_dot(file, child)


class Node(object):

    def __init__(self, type, label, agent, qaf, children):
        self.type = type
        self.label = label
        self.agent = agent
        self.qaf = qaf
        self.children = children

    def to_dot(self):
        if len(self.children) > 0:
            node_label = '\t"%s" [label="%s\\n%s"]' % (self.label, self.label, self.agent)
            dot_buffer = [node_label]

            for i in range(len(self.children)):
                child = self.children[i]

                if i == (len(self.children) - 1) / 2:
                    dot_buffer.append('"%s" -> "%s" [label="%s"];' % (self.label, child.label, self.qaf))
                else:
                    dot_buffer.append('"%s" -> "%s";' % (self.label, child.label))

            return "\n\t".join(dot_buffer) + "\n"
        else:
            node_shape = ',shape=box' if self.type == 'method' else ''
            return '\t"%s" [label="%s\\n%s"%s]\n' % (self.label, self.label, self.agent, node_shape)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        tree = Tree.load_from_taems(filename)
        filename = path.basename(filename)
        filename = 'graphs/' + filename
        tree.damp_to_dot(filename[:-6] + '.dot')
    else:
        print "Filename missing!"