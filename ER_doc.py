import networkx as nx
import random
import numpy as np
from abc import abstractmethod
import scipy.stats as stats

class Doctor:
    def __init__(self, Doc_id, agent_type, initial_opinion_ratio):
        self.agent_id = Doc_id
        self.opinion = 1
        self.agent_type = agent_type  

class Non_Doc:
    def __init__(self, NonDoc_id,  agent_type):
        self.agent_id = NonDoc_id
        lower, upper = 0, 1
        mu, sigma = 0.54, 0.29
        self.opinion = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        self.agent_type = agent_type  

class Bot:
    def __init__(self, Bot_id, agent_type):
        self.agent_id = Bot_id
        self.opinion = 0
        self.agent_type = agent_type  

class ERNetwork:
    def __init__(self, num_Doc, num_NonDoc, num_Bot, num_titled_doc, num_titled_bot, prob, initial_opinion_ratio, random_state) -> None:
        self.graph = nx.erdos_renyi_graph(num_Doc+num_NonDoc + num_Bot, prob, seed=random_state, directed=True)
        self.initial_opinion_ratio = initial_opinion_ratio
        self.agents = {}
        self.num_of_nodes = num_Doc + num_NonDoc + num_Bot
        self.degree = self.graph.degree
        self.nodes = []
        for node in self.graph.nodes():
            self.nodes.append(node)

        self.doc_nodes = random.sample(self.nodes, num_Doc)
        self.doc_nodes_title = random.sample(self.doc_nodes, num_titled_doc)
        self.bot_nodes = random.sample([node for node in self.nodes if node not in self.doc_nodes], num_Bot)
        self.bot_nodes_title = random.sample(self.bot_nodes, num_titled_bot)

        for node in self.graph.nodes():
            if not self.graph.has_edge(node, node):
                self.graph.add_edge(node, node)

            if node in self.doc_nodes:
                if node in self.doc_nodes_title:
                    agent = Doctor(node, "Doctor", initial_opinion_ratio)
                    self.agents[node] = agent
                else:
                    agent = Doctor(node, "Notitle_doctor", initial_opinion_ratio)
                    self.agents[node] = agent
                
            elif node in self.bot_nodes:
                if node in self.bot_nodes_title:
                    agent = Bot(node, "Titled_Bot")
                else:
                    agent = Bot(node, "Bot")
                self.agents[node] = agent

            else:
                agent = Non_Doc(node, "Non_Doc")
                self.agents[node] = agent
        
        self.calculate_edge_weights()
        self.weight_matrix = self.generate_weight_matrix()
        self.previous_opinions = [10 for node in self.graph.nodes()] 
    
    def calculate_edge_weights(self):

        for edge in self.graph.edges():
            node1, node2 = edge
            agent_type1 = self.agents[node1].agent_type
            agent_type2 = self.agents[node2].agent_type
            num_TitledNeighbors = 0
            num_NonTitledNeighbors = 0
            for neighbor in self.graph.neighbors(node1):
                if self.agents[neighbor].agent_type == "Doctor" or "Titled_Bot" and neighbor != node1:
                    num_TitledNeighbors = num_TitledNeighbors + 1
                elif self.agents[neighbor].agent_type == "Non_Doc" and neighbor != node1:
                    num_NonTitledNeighbors = num_NonTitledNeighbors + 1
                elif self.agents[neighbor].agent_type == "Notitle_doctor" and neighbor != node1:
                    num_NonTitledNeighbors = num_NonTitledNeighbors + 1
                elif self.agents[neighbor].agent_type == "Bot" and neighbor != node1:
                    num_NonTitledNeighbors = num_NonTitledNeighbors + 1

            # Assign weights based on agent types
            if agent_type1 == "Doctor" or agent_type1 == "Notitle_doctor" or agent_type1 == "Bot" or agent_type1 == "Titled_Bot":
                if node1 == node2:
                    weight = 1
                else:
                    weight = 0

            else:                
                if node1 != node2:
                    if agent_type2 =="Non_Doc" or agent_type2 == "Notitle_doctor" or agent_type2 == "Bot":
                        weight = 0.003/(0.183*num_TitledNeighbors + 0.003*num_NonTitledNeighbors + 0.906)
                    else:
                        weight = 0.183/(0.183*num_TitledNeighbors + 0.003*num_NonTitledNeighbors + 0.906)
                else:                         
                    weight = 0.906/(0.183*num_TitledNeighbors + 0.003*num_NonTitledNeighbors + 0.906)
        
            self.graph[node1][node2]['weight'] = weight
    
    def generate_weight_matrix(self):

        weight_matrix = np.zeros((self.num_of_nodes, self.num_of_nodes))

        for edge in self.graph.edges():
            node1, node2 = edge
            weight_matrix[node1][node2] = self.graph[node1][node2]['weight']

        return weight_matrix


    def update_opinions(self):
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            
            weighted_sum = sum(self.weight_matrix[node][n] * self.agents[n].opinion for n in neighbors)
            new_opinion = weighted_sum / sum(self.weight_matrix[node][n] for n in neighbors)
            self.agents[node].opinion = new_opinion

    def consensus_reached(self, tolerance=0.01):
        opinions = [self.agents[node].opinion for node in self.graph.nodes()]
        consensus_opinion = sum(opinions) / len(opinions)
        return all(abs(opinion - consensus_opinion) <= tolerance for opinion in opinions)
    
    def opinions_stabilized(self, tolerance=1e-4):
        current_opinions = [self.agents[node].opinion for node in self.graph.nodes()]
        deltas = [abs(curr - prev) for curr, prev in zip(current_opinions, self.previous_opinions)]
        max_delta = max(deltas)
        self.previous_opinions = current_opinions
        return max_delta < tolerance
