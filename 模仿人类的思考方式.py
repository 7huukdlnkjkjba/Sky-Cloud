import random
import time
from collections import defaultdict
import json
import os

class HumanLikeThinker:
    def __init__(self, knowledge_base=None, personality_traits=None):
        # 记忆系统（联想记忆）
        self.memory = defaultdict(list)
        self.knowledge_base = knowledge_base or self._default_knowledge()
        
        # 个性特征（影响决策）
        self.personality = personality_traits or {
            'openness': random.uniform(0.3, 0.9),
            'conscientiousness': random.uniform(0.3, 0.9),
            'extraversion': random.uniform(0.3, 0.9),
            'agreeableness': random.uniform(0.3, 0.9),
            'neuroticism': random.uniform(0.1, 0.7)
        }
        
        # 当前情感状态
        self.emotional_state = {
            'happiness': 0.7,
            'sadness': 0.1,
            'anger': 0.1,
            'fear': 0.1,
            'surprise': 0.1
        }
        
        # 思考日志
        self.thought_log = []
        
    def _default_knowledge(self):
        """默认知识库"""
        return {
            'general': {
                'cats': ['furry', 'independent', 'playful'],
                'dogs': ['loyal', 'friendly', 'energetic'],
                'work': ['responsibility', 'colleagues', 'salary'],
                'vacation': ['relax', 'travel', 'fun']
            },
            'personal': {
                'memories': ['childhood home', 'school friends', 'first job']
            }
        }
    
    def _associate(self, concept):
        """联想记忆 - 根据概念检索相关记忆和知识"""
        associations = []
        
        # 从知识库中检索
        for category in self.knowledge_base:
            if concept in self.knowledge_base[category]:
                associations.extend(self.knowledge_base[category][concept])
        
        # 从记忆中检索
        if concept in self.memory:
            associations.extend(self.memory[concept])
        
        # 添加一些随机联想（模拟人类思维的跳跃性）
        if random.random() < 0.3:  # 30%几率产生随机联想
            all_concepts = list(self.knowledge_base['general'].keys()) + \
                          list(self.knowledge_base['personal'].keys())
            random_concept = random.choice(all_concepts)
            if random_concept != concept:
                associations.append(f"randomly thought of {random_concept}")
        
        return associations if associations else ["no immediate associations"]
    
    def _emotional_influence(self, thought_process):
        """情感对思考过程的影响"""
        # 快乐增加创造性和乐观
        if self.emotional_state['happiness'] > 0.7:
            thought_process.append("feeling creative and optimistic")
        
        # 悲伤增加反思和谨慎
        if self.emotional_state['sadness'] > 0.5:
            thought_process.append("feeling reflective and cautious")
        
        # 愤怒导致冲动和简化思考
        if self.emotional_state['anger'] > 0.6:
            thought_process.append("feeling impulsive, simplifying thoughts")
            return thought_process[:max(2, len(thought_process)//2]
        
        # 恐惧导致过度分析和谨慎
        if self.emotional_state['fear'] > 0.6:
            thought_process.append("carefully analyzing all possibilities")
            thought_process.extend(["considering risks", "evaluating safety"])
        
        return thought_process
    
    def _logical_reasoning(self, question):
        """尝试逻辑推理"""
        reasoning = []
        
        # 简单的问题分解
        if 'why' in question.lower():
            reasoning.append("analyzing causes and effects")
        elif 'how' in question.lower():
            reasoning.append("considering processes and methods")
        
        # 基于个性的推理风格
        if self.personality['openness'] > 0.7:
            reasoning.append("exploring unconventional solutions")
        if self.personality['conscientiousness'] > 0.7:
            reasoning.append("systematically evaluating options")
        
        return reasoning
    
    def _intuitive_judgment(self):
        """直觉判断（基于经验和情感）"""
        intuition = []
        
        # 高开放性的人更依赖直觉
        if random.random() < self.personality['openness'] * 0.8:
            intuition.append("gut feeling suggests")
        
        # 当前情感状态影响直觉
        if self.emotional_state['happiness'] > 0.7:
            intuition.append("optimistic intuition")
        elif self.emotional_state['fear'] > 0.5:
            intuition.append("cautious intuition")
        
        return intuition
    
    def _update_emotional_state(self, thought_process):
        """根据思考内容更新情感状态"""
        content = ' '.join(thought_process).lower()
        
        # 正面词汇增加快乐
        positive_words = ['happy', 'joy', 'success', 'love', 'fun']
        if any(word in content for word in positive_words):
            self.emotional_state['happiness'] = min(1.0, self.emotional_state['happiness'] + 0.1)
        
        # 负面词汇增加悲伤或愤怒
        negative_words = ['sad', 'problem', 'angry', 'hate', 'risk']
        if any(word in content for word in negative_words):
            self.emotional_state['sadness'] = min(1.0, self.emotional_state['sadness'] + 0.1)
            self.emotional_state['anger'] = min(1.0, self.emotional_state['anger'] + 0.05)
        
        # 不确定词汇增加恐惧
        uncertain_words = ['worry', 'fear', 'danger', 'scared']
        if any(word in content for word in uncertain_words):
            self.emotional_state['fear'] = min(1.0, self.emotional_state['fear'] + 0.15)
    
    def think_about(self, concept, question=None):
        """模拟思考一个概念或问题的过程"""
        thought_process = []
        
        # 1. 感知和联想阶段
        thought_process.append(f"Thinking about {concept}...")
        associations = self._associate(concept)
        thought_process.append(f"Associations: {', '.join(associations[:3])}")
        
        # 2. 如果有问题，尝试逻辑推理
        if question:
            reasoning = self._logical_reasoning(question)
            thought_process.extend(reasoning)
        
        # 3. 情感影响
        thought_process = self._emotional_influence(thought_process)
        
        # 4. 直觉判断
        intuition = self._intuitive_judgment()
        if intuition:
            thought_process.append(f"Intuition: {' '.join(intuition)}")
        
        # 5. 决策/结论
        conclusion_options = [
            "This seems reasonable",
            "I'm not entirely sure",
            "Need more information",
            "This feels right",
            "I have doubts about this"
        ]
        # 个性影响结论
        if self.personality['neuroticism'] > 0.6:
            conclusion_options.extend(["I'm worried about this", "This might be risky"])
        if self.personality['agreeableness'] > 0.7:
            conclusion_options.extend(["This seems good for everyone", "I want everyone to be happy"])
        
        conclusion = random.choice(conclusion_options)
        thought_process.append(f"Conclusion: {conclusion}")
        
        # 更新情感状态
        self._update_emotional_state(thought_process)
        
        # 记录思考过程
        self.thought_log.append({
            'concept': concept,
            'question': question,
            'thoughts': thought_process,
            'timestamp': time.time()
        })
        
        # 模拟人类思考时间
        time.sleep(random.uniform(0.5, 2.0))
        
        return thought_process
    
    def save_memory(self, filename):
        """保存记忆到文件"""
        with open(filename, 'w') as f:
            json.dump({
                'memory': dict(self.memory),
                'knowledge_base': self.knowledge_base,
                'personality': self.personality,
                'thought_log': self.thought_log
            }, f, indent=2)
    
    def load_memory(self, filename):
        """从文件加载记忆"""
        if os.path.exists(filename):
            with open(filename) as f:
                data = json.load(f)
                self.memory = defaultdict(list, data.get('memory', {}))
                self.knowledge_base = data.get('knowledge_base', self._default_knowledge())
                self.personality = data.get('personality', self.personality)
                self.thought_log = data.get('thought_log', [])
            return True
        return False


# 示例使用
if __name__ == "__main__":
    print("Initializing human-like thinker...")
    thinker = HumanLikeThinker()
    
    # 模拟思考过程
    topics = ["cats", "work", "vacation", "future", "AI"]
    questions = [
        None,
        "Why do I need to work?",
        "How to plan the perfect vacation?",
        "What will the future hold?",
        "Will AI replace human jobs?"
    ]
    
    for topic, question in zip(topics, questions):
        print(f"\n=== Thinking about: {topic} {f'({question})' if question else ''} ===")
        thoughts = thinker.think_about(topic, question)
        for thought in thoughts:
            print(f"- {thought}")
    
    # 保存记忆
    thinker.save_memory('thinker_memory.json')
    print("\nMemory saved to thinker_memory.json")