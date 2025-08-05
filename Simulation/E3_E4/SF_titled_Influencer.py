import networkx as nx
import random
import numpy as np
from abc import abstractmethod
import scipy.stats as stats

class Doctor:
    def __init__(self, Doc_id, agent_type):
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

class Influencer:
    def __init__(self, Doc_id, agent_type):
        self.agent_id = Doc_id
        self.opinion = 1
        self.agent_type = agent_type        

class SFNetwork:

    def __init__(self, num_Doc, num_NonDoc, num_Influencer, num_stub_influencer, num_titled_doc, titled_influencer, prob, random_state) -> None:
        # self.graph = nx.barabasi_albert_graph(num_Doc+num_NonDoc, prob, seed = random_state, directed = True)
        self.graph = nx.DiGraph(nx.scale_free_graph(num_Doc+num_NonDoc, seed = random_state))
        self.agents = {}
        self.num_of_nodes = num_Doc+num_NonDoc
        self.degree = self.graph.in_degree()
        self.nodes = []
        for node in self.graph.nodes():
            self.nodes.append(node)
        
        influencer = sorted(self.degree, key=lambda x: x[1], reverse=True)[:num_Influencer]
        self.influencer = [item[0] for item in influencer]
        low_degree = [x for x in self.nodes if x not in self.influencer]


        self.doc_nodes = random.sample(low_degree, num_Doc)
        self.doc_nodes_title = random.sample(self.doc_nodes, num_titled_doc)

        stub_influencer = sorted(self.degree, key=lambda x: x[1], reverse=True)[:num_stub_influencer]
        self.stub_influencer = [item[0] for item in stub_influencer]

        # titled_influencer = sorted(self.degree, key=lambda x: x[1], reverse=True)[:num_titled_influencer]
        # self.titled_influencer = [item[0] for item in titled_influencer]


        for node in self.graph.nodes():
            if not self.graph.has_edge(node, node):
                self.graph.add_edge(node, node)
            if node in self.doc_nodes:
                if node in self.doc_nodes_title:
                    agent = Doctor(node, "Doctor")
                    self.agents[node] = agent
                else:
                    agent = Doctor(node, "Notitle_doctor")
                    self.agents[node] = agent
            elif node in self.stub_influencer:
                if titled_influencer == False:
                    agent = Influencer(node, "Influencer")
                    self.agents[node] = agent
                else:
                    agent = Influencer(node, "Titled_Influencer")
                    self.agents[node] = agent
            else:
                agent = Non_Doc(node, "Non_Doc")
                self.agents[node] = agent
        
        self.calculate_edge_weights()
        self.weight_matrix = self.generate_weight_matrix() 
    
    def calculate_edge_weights(self):

        for edge in self.graph.edges():
            node1, node2 = edge
            agent_type1 = self.agents[node1].agent_type
            agent_type2 = self.agents[node2].agent_type
            num_DocNeighbors = 0
            num_NonDocNeighbors = 0
            num_stub_Influencer = 0
            num_titled_Influencer = 0
            for neighbor in self.graph.neighbors(node1):
                if self.agents[neighbor].agent_type == "Doc" and neighbor != node1:
                    num_DocNeighbors = num_DocNeighbors + 1
                elif self.agents[neighbor].agent_type == "Non_Doc" and neighbor != node1:
                    num_NonDocNeighbors = num_NonDocNeighbors + 1
                elif self.agents[neighbor].agent_type == "Notitle_doctor" and neighbor != node1:
                    num_NonDocNeighbors = num_NonDocNeighbors + 1
                elif self.agents[neighbor].agent_type == "Influencer" and neighbor != node1:
                    num_stub_Influencer = num_stub_Influencer + 1 
                elif self.agents[neighbor].agent_type == "Titled_Influencer" and neighbor != node1:
                    num_stub_Influencer = num_titled_Influencer + 1   
            if agent_type1 == "Doctor" or agent_type1 == "Notitle_doctor" or agent_type1 == "Influencer" or agent_type1 == "Titled_Influencer":
                if node1 == node2:
                    weight = 1
                # elif agent_type2 == "Doctor" and num_DocNeighbors !=0:
                #     weight = 0.4/num_DocNeighbors
                else:
                    weight = 0
            else:  
            # Assign weights based on agent types
                if node1 != node2:
                    if agent_type2 == "Non_Doc" or agent_type2 == "Notitle_doctor" or agent_type2 == "Influencer":
                        weight = 0.003/(0.183*(num_DocNeighbors + num_titled_Influencer) + 0.003*(num_NonDocNeighbors+num_stub_Influencer) + 0.906)
                    else:
                        weight = 0.183/(0.183*(num_DocNeighbors + num_titled_Influencer) + 0.003*(num_NonDocNeighbors+num_stub_Influencer) + 0.906)
                else:                         
                    weight = 0.906/(0.183*(num_DocNeighbors + num_titled_Influencer) + 0.003*(num_NonDocNeighbors+num_stub_Influencer) + 0.906)
        
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