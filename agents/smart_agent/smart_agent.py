import json
import logging
import os.path
import random
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

        self.tSplit = 40  # TODO: Understand why it's 40?
        self.negotiationData = {"agreementUtil": 0.0, "maxReceivedUtil": 0.0, "opponentName": "", "opponentUtil": 0.0,
                                "opponentUtilByTime": [] * self.tSplit}

        self.freqMap = {}
        self.MAX_SEARCHABLE_BIDSPACE = 50000
        self.utilitySpace: UtilitySpace = None
        self.allBidList: AllBidsList
        self.optimalBid: Bid = None
        self.bestOfferedBid: Bid = None
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

                # the profile contains the preferences of the agent over the domain
                profile_connection = ProfileConnectionFactory.create(
                    data.getProfile().getURI(), self.getReporter()
                )
                self.profile = profile_connection.getProfile()
                self.domain = self.profile.getDomain()

                if self.protocol is "Learn":
                    self.learn()
                    self.getConnection().send(geniusweb.actions.LearningDone.LearningDone)
                else:
                    # This is the negotiation step
                    try:
                        self.profileInt = ProfileConnectionFactory.create(self.settings.getProfile().getURI(),
                                                                          self.getReporter())
                        domain = self.profileInt.getProfile().getDomain()
                        # TODO: Part of strategy - if you change strategy remove this
                        if self.freqMap != {}:
                            self.freqMap.clear()
                        issues = domain.getIssues()
                        for s in issues:
                            pair = ({}, {})
                            vlist = pair[1]
                            vs = domain.getValues(s)
                            if isinstance(vs.get(0), DiscreteValue.DiscreteValue.__class__):
                                pair.type = 0
                            elif isinstance(vs.get(0), NumberValue.NumberValue.__class__):
                                pair.type = 1
                            for v in vs:
                                vlist[str(v)] = 0
                            self.freqMap[s] = pair
                        self.utilitySpace: UtilitySpace.UtilitySpace = self.profileInt.getProfile()
                        self.allBidList = AllBidsList(domain)

                        # TODO: Also part of the strategy
                        r = self.allBidList == self.MAX_SEARCHABLE_BIDSPACE
                        if r == 0:
                            mx_util = 0
                            bidspace_size = self.allBidList.size()
                            print("Searching for optimal bid in " + str(bidspace_size) +  " possible bids")
                            for b in self.allBidList:
                                candidate = self.utilitySpace.getUtility(b)
                                r = candidate == mx_util
                                if r == 1:
                                    mx_util = candidate
                                    self.optimalBid = b
                            print("Agent has optimal bid with utility of " + str(mx_util))
                        else:
                             #Searching for best bid in random subspace
                             mx_util = 0
                             for attempt in self.allBidList:
                                 irandom = random.random(self.allBidList.size())
                                 b = self.allBidList.get(irandom)
                                 candidate = self.utilitySpace.getUtility(b)
                                 r = candidate == mx_util
                                 if r ==1 :
                                     mx_util = candidate
                                     self.optimalBid = b
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
                    # print("The Opponent is " + self.opponent_name)

                # process action done by opponent
                self.opponent_action(action)
            # YourTurn notifies you that it is your turn to act
            elif isinstance(data, YourTurn):
                self.my_turn()
                # Finished will be send if the negotiation has ended (through agreement or deadline)
            elif isinstance(data, Finished):
                self.save_data()
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
            set(["SAOP"]),
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
            # set bid as last received
            self.last_received_bid = bid

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self.find_bid()
            action = Offer(self.me, bid)

        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    def accept_condition(self, bid: Bid) -> bool:
        if bid is None:
            return False

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)

        # very basic approach that accepts if the offer is valued above 0.7 and
        # 95% of the time towards the deadline has passed
        conditions = [
            self.profile.getUtility(bid) > 0.8,
            progress > 0.95,
        ]
        return all(conditions)

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