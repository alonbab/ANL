import json
import logging
import math
import os.path
import random
from decimal import Decimal
from random import randint
from time import time
from typing import cast
from typing import final

import geniusweb.actions.LearningDone
from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue import DiscreteValue, NumberValue
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace import UtilitySpace
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.profileconnection.ProfileInterface import (
    ProfileInterface
)
from geniusweb.progress.ProgressRounds import ProgressRounds
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from agents.template_agent.utils.opponent_model import OpponentModel


class SmartAgent(DefaultParty):
    def __init__(self):
        super().__init__()


        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.profileInt: ProfileInterface = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.random: final(random) = random.Random()
        self.protocol = ""
        self.opponent_name: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.tSplit = 40
        self.tPhase = 0.2
        self.newWeight = 0.3
        self.smoothWidth = 3
        self.opponentDecrease = 0.65
        self.defaultAlpha = 10.7
        self.alpha = self.defaultAlpha

        self.opponent_avgUtility = 0.0
        self.opponent_negotiations = 0
        self.opponent_avgMaxUtility = {}
        self.opponent_encounters = {}

        self.stdUtility = 0.0
        self.negotiationResults = []
        self.avgOpponentUtility = {}
        self.opponentAlpha = {}
        self.opponent_sum = [0.0] * 5000
        self.opponent_Counter = [0.0] * 5000

        self.persistentState = {"opponentAlpha": self.defaultAlpha, "AvgMaxUtility": 0.0}
        self.negotiationData = {"agreementUtil": 0.0, "maxReceivedUtil": 0.0, "opponentName": "", "opponentUtil": 0.0,
                                "opponentUtilByTime": [0.0] * self.tSplit}
        self.opponentUtilityByTime = self.negotiationData["opponentUtilByTime"]
        self.need_to_read_persistent_data = True
        self.freqMap = {}
        self.MAX_SEARCHABLE_BIDSPACE = 50000
        self.utilitySpace: UtilitySpace = None
        self.allBidList: AllBidsList
        self.optimalBid: Bid = None
        self.bestOfferedBid: Bid = None
        self.utilThreshold = None

        self.opThreshold = None

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.logger.log(logging.INFO, "party is initialized")

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        try:
            if isinstance(data, Settings):
                # data is an object that is passed at the start of the negotiation
                self.settings = cast(Settings, data)
                # ID of my agent
                self.me = self.settings.getID()

                # progress towards the deadline has to be tracked manually through the use of the Progress object
                self.progress = self.settings.getProgress()

                self.protocol = self.settings.getProtocol().getURI().getPath()
                self.parameters = self.settings.getParameters()
                self.storage_dir = self.parameters.get("storage_dir")

                # TODO: Add persistance
                # the profile contains the preferences of the agent over the domain
                profile_connection = ProfileConnectionFactory.create(
                    data.getProfile().getURI(), self.getReporter()
                )
                self.profile = profile_connection.getProfile()
                self.domain = self.profile.getDomain()

                if str(self.settings.getProtocol().getURI()) == "Learn":
                    self.learn()
                    self.getConnection().send(geniusweb.actions.LearningDone.LearningDone)
                else:
                    # This is the negotiation step
                    try:
                        self.profileInt = ProfileConnectionFactory.create(self.settings.getProfile().getURI(),
                                                                          self.getReporter())
                        domain = self.profileInt.getProfile().getDomain()
                        # TODO: Part of strategy - if you change strategy remove this
                        # if self.freqMap != {}:
                        #     self.freqMap.clear()
                        # issues = domain.getIssues()
                        # for s in issues:
                        #     pair = ({}, {})
                        #     vlist = pair[1]
                        #     vs = domain.getValues(s)
                        #     if isinstance(vs.get(0), DiscreteValue.DiscreteValue.__class__):
                        #         pair.type = 0
                        #     elif isinstance(vs.get(0), NumberValue.NumberValue.__class__):
                        #         pair.type = 1
                        #     for v in vs:
                        #         vlist[str(v)] = 0
                        #     self.freqMap[s] = pair
                        self.utilitySpace: UtilitySpace.UtilitySpace = self.profileInt.getProfile()
                        self.allBidList = AllBidsList(domain)

                        # TODO: Also part of the strategy
                        r = self.allBidList == self.MAX_SEARCHABLE_BIDSPACE
                        if r == 0 or r == -1:
                            mx_util = 0
                            bidspace_size = self.allBidList.size()
                            print("Searching for optimal bid in " + str(bidspace_size) + " possible bids")
                            for i in range(0, bidspace_size, 1):
                                b: Bid = self.allBidList.get(i)
                                candidate = self.utilitySpace.getUtility(b)
                                r = candidate > mx_util
                                if r == 1:
                                    mx_util = candidate
                                    self.optimalBid = b
                            print("Agent has optimal bid with utility of " + str(mx_util))
                        else:
                            # Searching for best bid in random subspace
                            mx_util = 0
                            for attempt in self.allBidList:
                                irandom = random.random(self.allBidList.size())
                                b = self.allBidList.get(irandom)
                                candidate = self.utilitySpace.getUtility(b)
                                r = candidate > mx_util
                                if r == 1:
                                    mx_util = candidate
                                    self.optimalBid = b
                            print("agent has best (perhaps sub optimal) bid with utility of " + str(mx_util))
                    except:
                        raise Exception("Illegal state exception")
                profile_connection.close()
            # ActionDone informs you of an action (an offer or an accept)
            # that is performed by one of the agents (including yourself).
            elif isinstance(data, ActionDone):
                action = cast(ActionDone, data).getAction()
                actor = action.getActor()
                # ignore action if it is our action
                if actor != self.me:
                    # obtain the name of the opponent, cutting of the position ID.
                    self.opponent_name = str(actor).rsplit("_", 1)[0]
                    # TODO: Add persistance
                    if self.need_to_read_persistent_data:
                        self.negotiationData = self.readPersistentNegotiationData()
                        self.need_to_read_persistent_data = False
                    self.negotiationData["opponentName"] = self.opponent_name
                    self.opThreshold = self.getSmoothThresholdOverTime(self.opponent_name)
                    if self.opThreshold != None:
                        for i in range(1, self.tSplit, 1):
                            if self.opThreshold[i] < 0:
                                self.opThreshold[i] = self.opThreshold[i-1]
                    self.alpha = self.persistentState["opponentAlpha"]
                    print("alpha is " + str(self.persistentState["opponentAlpha"]))
                    if self.alpha < 0.0:
                        self.alpha = self.defaultAlpha
                    self.updateNegotiationData()

                # process action done by opponent
                    self.opponent_action(action)

            # YourTurn notifies you that it is your turn to act
            elif isinstance(data, YourTurn):
                if isinstance(self.progress,ProgressRounds):
                    self.progress = cast(ProgressRounds, self.progress).advance()
                self.my_turn()
                # Finished will be send if the negotiation has ended (through agreement or deadline)
            elif isinstance(data, Finished):
                # terminate the agent MUST BE CALLED
                self.logger.log(logging.INFO, "party is terminating:")
                super().terminate()
            else:
                self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))
        except:
            raise Exception("Illegal state exception")

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP", "Learn"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Template agent for the ANL 2022 competition"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()
            # update opponent model with bid
            self.opponent_model.update(bid)
            self.updateNegotiationData()
            # set bid as last received
            self.last_received_bid = bid
            utilVal = self.utilitySpace.getUtility(bid)
            self.negotiationData["maxReceivedUtil"] = utilVal
            print( self.negotiationData["maxReceivedUtil"])

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        if self.isNearNegotiationEnd() > 0 :
            index = int((self.tSplit - 1)/(1-self.tPhase)*(self.progress.get(time()*1000) - self.tPhase))
            if self.opponent_sum[index]:
                self.opponent_sum[index] = self.calcOpValue(self.last_received_bid)
            else:
                self.opponent_sum[index] += self.calcOpValue(self.last_received_bid)
            self.opponent_Counter[index] += 1
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid: Bid = None

            if self.bestOfferedBid == None:
                self.bestOfferedBid = self.last_received_bid
            elif self.utilitySpace.getUtility(self.last_received_bid) > self.utilitySpace.getUtility(self.bestOfferedBid):
                self.bestOfferedBid = self.last_received_bid
            if self.isNearNegotiationEnd() == 0:
                for attempt in range(0,1000,1):
                    if not self.accept_condition(bid):
                        i = random.randint(0, self.allBidList.size())
                        bid = self.allBidList.get(i)
                    if self.accept_condition(bid):
                        bid = bid
                    else:
                        bid = self.optimalBid
            if self.isNearNegotiationEnd() == 1 or self.isNearNegotiationEnd() == 2:
                for attempt in range(0,1000,1):
                    if bid != self.optimalBid and not self.accept_condition(bid) and not self.isOpGood(bid):
                        i = random.randint(0, self.allBidList.size())
                        bid = self.allBidList.get(i)
                    if self.progress.get(time()*1000) > 0.99 and self.accept_condition((self.bestOfferedBid)):
                        bid = self.bestOfferedBid
                    if not self.accept_condition(bid):
                        bid = self.optimalBid
            action = Offer(self.me, bid)
        # send the action
        self.send_action(action)

    def readPersistentNegotiationData(self):
        if os.path.exists(f"{self.storage_dir}/{self.opponent_name}"):
            with open(f"{self.storage_dir}/{self.opponent_name}", "r") as f:
                data = json.load(f)
                return data
        else:
            return {"opponentAlpha": self.defaultAlpha, "agreementUtil": 0.0, "maxReceivedUtil": 0.0, "opponentName": self.opponent_name,
                    "opponentUtil": 0.0,
                    "opponentUtilByTime": [0.0] * self.tSplit}

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    def isNearNegotiationEnd(self):
         if self.progress.get(time()*1000) < self.tPhase:
             return 0
         else:
             return 1

    def calcOpValue(self, bid: Bid):
        if not bid:
            return 0
        # own_utility = self.profile.getProfile().getUtility(bid)
        opponent_utility = self.opponent_model.get_predicted_utility(bid) #.getUtility(bid)
        return opponent_utility

    def isOpGood(self,bid: Bid):
        if bid == None:
            return 0
        value = self.calcOpValue(bid)
        index = int(((self.tSplit-1)/(1-self.tPhase)*(self.progress.get(time()*1000) - self.tPhase)))
        if self.opThreshold != None:
            self.opThreshold = max(1-2*self.opThreshold[index],0.2)
        else:
            self.opThreshold = 0.6
        return value > self.opThreshold
    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    def accept_condition(self, bid: Bid) -> bool:
        if bid is None or self.opponent_name is None:
            return False
        avgMaxUtility = self.avgOpponentUtility[self.opponent_name]
        if self.optimalBid != None:
            maxValue = 0.95 * float(self.utilitySpace.getUtility(self.optimalBid))
        else:
            maxValue = 0.95
        if self.isKnownOpponent(self.opponent_name):
            avgMaxUtility = self.avgOpponentUtility[self.opponent_name]
        if self.alpha != 0:
         self.utilThreshold = maxValue - (maxValue - 0.6*self.opponent_avgUtility - 0.4*avgMaxUtility + pow(self.stdUtility, 2)) * (math.exp(self.alpha*self.progress.get(time()*1000) - 1) - 1)/(math.exp(self.alpha) - 1)
        return  self.utilitySpace.getUtility(bid) >= self.utilThreshold

    def find_bid(self) -> Bid:
        # compose a list of all possible bids
        domain = self.profile.getDomain()
        all_bids = AllBidsList(domain)

        best_bid_score = 0.0
        best_bid = None

        # take 500 attempts to find a bid according to a heuristic score
        for _ in range(500):
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            bid_score = self.score_bid(bid)
            if bid_score > best_bid_score:
                best_bid_score, best_bid = bid_score, bid

        return best_bid

    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.1) -> float:
        """Calculate heuristic score for a bid

        Args:
            bid (Bid): Bid to score
            alpha (float, optional): Trade-off factor between self interested and
                altruistic behaviour. Defaults to 0.95.
            eps (float, optional): Time pressure factor, balances between conceding
                and Boulware behaviour over time. Defaults to 0.1.

        Returns:
            float: score
        """
        progress = self.progress.get(time() * 1000)

        our_utility = float(self.profile.getUtility(bid))

        time_pressure = 1.0 - progress ** (1 / eps)
        score = alpha * time_pressure * our_utility

        if self.opponent_model is not None:
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            opponent_score = (1.0 - alpha * time_pressure) * opponent_utility
            score += opponent_score

        return score

    def learn(self):
     return "ok"

    def isKnownOpponent(self, opponentName):
         return self.opponent_encounters.get(opponentName, 0)

    def getSmoothThresholdOverTime(self, opponentName):
          if not self.isKnownOpponent(opponentName):
              return None
          opponentTimeUtil = self.negotiationData["opponentUtilByTime"]
          smoothedTimeUtil = [0.0]*self.tSplit

          for i in range(0,self.tSplit, 1):
              for j in range(max(i-self.smoothWidth,0), min(i+self.smoothWidth+1,self.tSplit), 1):
                  smoothedTimeUtil[i] += opponentTimeUtil[j]
              smoothedTimeUtil[i] /= (min(i+self.smoothWidth+1, self.tSplit) - max(i-self.smoothWidth, 0))
          return smoothedTimeUtil
    def calcAlpha(self, opponentName):
        alphaArray = self.getSmoothThresholdOverTime(opponentName)
        if alphaArray == None:
            return self.defaultAlpha
        for maxIndex in range(0, self.tSplit, 1):
            if alphaArray[maxIndex] >0.2:
                break
        maxValue = alphaArray[0]
        minValue = alphaArray[max(maxIndex - self.smoothWidth - 1, 0)]

        if maxValue - minValue < 0.1:
            return self.defaultAlpha
        for t in range(0, maxIndex, 1):
            if alphaArray[t] > (maxValue-self.opponentDecrease*(maxValue-minValue)):
                break
        calibratedPolynom = {572.83,-1186.7, 899.29, -284.68, 32.911}
        alpha = calibratedPolynom[0]

        # lowers utility at 85% of the time why 85% ???
        tTime = self.tPhase + (1-self.tPhase)*(maxIndex*(float(t)/self.tSplit) + (self.tSplit-maxIndex)*0.85)/self.tSplit
        for i in range(1, len(calibratedPolynom), 1):
            alpha = alpha*tTime + calibratedPolynom[i]

        print("Alpha is :" + str(alpha))
        return alpha

    def updateNegotiationData(self):
       if self.negotiationData.get("agreementUtil") > 0:
           newUtil = self.negotiationData.get("agreementUtil")
       else:
           newUtil = self.opponent_avgUtility - 1.1 * math.pow(self.stdUtility, 2)
       self.opponent_avgUtility = (self.opponent_avgUtility * self.opponent_negotiations + newUtil) / (
                    self.opponent_negotiations + 1)
       self.opponent_negotiations += 1
       self.avgOpponentUtility[self.opponent_name] = self.opponent_avgUtility
       self.negotiationResults.append(self.negotiationData["agreementUtil"])
       self.stdUtility = 0.0
       for util in self.negotiationResults:
           self.stdUtility += math.pow(util - self.opponent_avgUtility, 2)
       self.stdUtility = math.sqrt(self.stdUtility / self.opponent_negotiations)

       opponentName = self.negotiationData["opponentName"]
       print(opponentName)

       if opponentName != "":
            if self.opponent_encounters.get(opponentName):
                encounters = self.opponent_encounters.get(opponentName)
            else:
                encounters = 0
            self.opponent_encounters[opponentName] = encounters + 1

            if self.opponent_avgMaxUtility.get(opponentName):
                avgUtil = self.opponent_avgMaxUtility[opponentName]
            else:
                avgUtil = 0.0
            calculated_opponent_avg_max_utility = (float(avgUtil * encounters) + float(self.negotiationData["maxReceivedUtil"])) / (
                    encounters + 1)
            self.opponent_avgMaxUtility.get(opponentName, calculated_opponent_avg_max_utility)

            if self.avgOpponentUtility[opponentName]:
                avgOpUtil = self.avgOpponentUtility[opponentName]
            else:
                avgOpUtil = 0.0
            calculated_opponent_avg_utility = (float(avgOpUtil * encounters) + float(self.negotiationData["opponentUtil"])) / (
                    encounters + 1)
            self.avgOpponentUtility[opponentName] = calculated_opponent_avg_utility
            if self.opponentUtilityByTime:
                opponentTimeUtility = self.opponentUtilityByTime
            else:
                opponentTimeUtility = [0.0] * self.tSplit

            newUtilData = self.negotiationData.get("opponentUtilByTime")
            if opponentTimeUtility[0] > 0.0:
                ratio = ((1 - self.newWeight) * opponentTimeUtility[0] + self.newWeight * newUtilData[0] /
                         opponentTimeUtility[0])
            else:
                ratio = 1
            for i in range(0, self.tSplit, 1):
                if newUtilData[i] > 0:
                    valueUtilData = (
                                (1 - self.newWeight) * opponentTimeUtility[i] + self.newWeight * newUtilData[i])
                    opponentTimeUtility[i] = valueUtilData
                else:
                    opponentTimeUtility[i] *= ratio
            self.negotiationData["opponentUtilByTime"] = opponentTimeUtility
            print(self.negotiationData["opponentUtilByTime"])
            self.opponentAlpha[opponentName] = self.calcAlpha(opponentName)


