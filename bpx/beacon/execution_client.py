from __future__ import annotations

import logging
import asyncio
import time

from typing import (
    Optional,
    Union,
)

from web3 import Web3, HTTPProvider
from web3.method import Method
from web3.module import Module
from web3.providers.rpc import URI
import jwt
from hexbytes import HexBytes

from bpx.util.path import path_from_root
from bpx.consensus.block_record import BlockRecord
from bpx.types.blockchain_format.sized_bytes import bytes20, bytes32, bytes256
from bpx.util.ints import uint64, uint256
from bpx.types.blockchain_format.execution_payload import ExecutionPayloadV2, WithdrawalV1
from bpx.util.byte_types import hexstr_to_bytes
from bpx.consensus.block_rewards import create_withdrawals

COINBASE_NULL = bytes20.fromhex("0000000000000000000000000000000000000000")
BLOCK_HASH_NULL = bytes32.fromhex("0000000000000000000000000000000000000000000000000000000000000000")

log = logging.getLogger(__name__)

class HTTPAuthProvider(HTTPProvider):
    secret: bytes

    def __init__(
        self,
        secret: bytes,
        endpoint_uri: Optional[Union[URI, str]] = None,
    ) -> None:
        self.secret = secret
        super().__init__(endpoint_uri)
    
    def get_request_headers(self) -> Dict[str, str]:
        headers = super().get_request_headers()
        
        encoded_jwt = jwt.encode(
            {
                "iat": int(time.time())
            },
            self.secret,
            algorithm="HS256"
        )
        
        headers.update(
            {
                "Authorization": "Bearer " + encoded_jwt
            }
        )
        return headers

class EngineModule(Module):
    exchange_transition_configuration_v1 = Method("engine_exchangeTransitionConfigurationV1")
    forkchoice_updated_v2 = Method("engine_forkchoiceUpdatedV2")
    get_payload_v2 = Method("engine_getPayloadV2")
    new_payload_v2 = Method("engine_newPayloadV2")

class ExecutionClient:
    beacon: Beacon
    w3: Web3
    peak_txb_hash: Optional[bytes32]
    payload_id: Optional[str]
    syncing: bool

    def __init__(
        self,
        beacon,
    ):
        self.beacon = beacon
        self.w3 = None
        self.peak_txb_hash = None
        self.payload_id = None
        self.syncing = False
        self.connected = False
        
    
    async def exchange_transition_configuration_task(self):
        log.debug("Starting exchange transition configuration loop")

        while True:
            try:
                self._ensure_web3_init()
                try:
                    self.w3.engine.exchange_transition_configuration_v1({
                        "terminalTotalDifficulty": "0x0",
                        "terminalBlockHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
                        "terminalBlockNumber": "0x0"
                    })
                except:
                    self.connected = False
                    raise
                self.connected = True
                log.info("Exchanged transition configuration with execution client")
            except Exception as e:
                log.error(f"Exception in exchange transition configuration: {e}")
                
            await asyncio.sleep(60)
    
    
    async def new_payload(
        self,
        payload: ExecutionPayloadV2,
    ) -> str:
        self._ensure_web3_init()
        
        raw_transactions = []
        for transaction in payload.transactions:
            raw_transactions.append("0x" + transaction.hex())
        
        raw_withdrawals = []
        for withdrawal in payload.withdrawals:
            raw_withdrawals.append({
                "index": Web3.to_hex(withdrawal.index),
                "validatorIndex": Web3.to_hex(withdrawal.validatorIndex),
                "address": "0x" + withdrawal.address.hex(),
                "amount": Web3.to_hex(withdrawal.amount),
            })
        
        raw_payload = {
            "parentHash": "0x" + payload.parentHash.hex(),
            "feeRecipient": "0x" + payload.feeRecipient.hex(),
            "stateRoot": "0x" + payload.stateRoot.hex(),
            "receiptsRoot": "0x" + payload.receiptsRoot.hex(),
            "logsBloom": "0x" + payload.logsBloom.hex(),
            "prevRandao": "0x" + payload.prevRandao.hex(),
            "blockNumber": Web3.to_hex(payload.blockNumber),
            "gasLimit": Web3.to_hex(payload.gasLimit),
            "gasUsed": Web3.to_hex(payload.gasUsed),
            "timestamp": Web3.to_hex(payload.timestamp),
            "extraData": "0x" + payload.extraData.hex(),
            "baseFeePerGas": Web3.to_hex(payload.baseFeePerGas),
            "blockHash": "0x" + payload.blockHash.hex(),
            "transactions": raw_transactions,
            "withdrawals": raw_withdrawals,
        }

        try:
            result = self.w3.engine.new_payload_v2(raw_payload)
        except:
            self.connected = False
            raise
        self.connected = True
        if result.validationError is not None:
            log.error(
                f"Payload validation error: Eheight={payload.blockNumber}, Ehash={payload.blockHash}, "
                f"status={result.status}, error={result.validationError}"
            )
        else:
            log.info(
                f"Processed execution payload: Eheight={payload.blockNumber}, Ehash={payload.blockHash}, "
                f"status={result.status}"
            )
        
        return result.status
    
    
    async def forkchoice_update(
        self,
        block: BlockRecord,
    ) -> None:
        log.info("Fork choice update")
        
        self.payload_id = None
        
        self._ensure_web3_init()
        
        head_ehash = block.execution_block_hash
        log.info(f" |- Head: Bheight={block.height}, Bhash={block.header_hash}, Ehash={head_ehash}")
        
        safe_ehash = head_ehash
        log.info(f" |- Safe: Bheight={block.height}, Bhash={block.header_hash}, Ehash={safe_ehash}")
        
        final_bheight: Optional[uint64]
        final_bhash: bytes32
        final_ehash: bytes32
        sub_slots = 0
        curr = block
        while True:
            if curr.first_in_sub_slot:
                sub_slots += 1
            
            final_bheight = curr.height
            final_bhash = curr.header_hash
            final_ehash = curr.execution_block_hash
            
            if sub_slots == 2:
                break
            
            if curr.prev_transaction_block_hash == self.beacon.constants.GENESIS_CHALLENGE:
                final_bheight = None
                final_bhash = None
                final_ehash = BLOCK_HASH_NULL
                break
            
            curr = await self.beacon.blockchain.get_block_record_from_db(curr.prev_transaction_block_hash)
        log.info(f" |- Finalized: Bheight={final_bheight}, Bhash={final_bhash}, Ehash={final_ehash}")
        
        forkchoice_state = {
            "headBlockHash": "0x" + head_ehash.hex(),
            "safeBlockHash": "0x" + safe_ehash.hex(),
            "finalizedBlockHash": "0x" + final_ehash.hex(),
        }
        payload_attributes = None
        
        synced = self.beacon.sync_store.get_sync_mode() is False
        if synced:
            coinbase = self.beacon.config["coinbase"]
            if bytes20.from_hexstr(coinbase) == COINBASE_NULL:
                log.warning("Coinbase is not set! Payload will not be built and farming is not possible.")
            else:
                payload_attributes = self._create_payload_attributes(block, coinbase)

        try:
            result = self.w3.engine.forkchoice_updated_v2(forkchoice_state, payload_attributes)
        except:
            self.connected = False
            raise
        self.connected = True
        
        self.peak_txb_hash = block.header_hash
        
        if result.payloadStatus.validationError is not None:
            log.error(
                f"Fork choice not updated: status={result.payloadStatus.status}, "
                f"validation error: {result.payloadStatus.validationError}"
            )
        else:
            log.info(
                f"Fork choice updated: status={result.payloadStatus.status}"
            )
        
        if result.payloadId is not None:
            self.payload_id = result.payloadId
            log.info(f"Payload building started: payload_id={self.payload_id}")
        elif synced:
            log.warning("Payload building not started")
        
        if result.payloadStatus.status == "VALID" and self.syncing:
            self.syncing = False
            log.info(f"Execution Client is now fully synced")
        elif result.payloadStatus.status != "VALID" and not self.syncing:
            self.syncing = True
            log.info(f"Execution Client syncing started")
        
        return result.payloadStatus.status
    
    
    def get_payload(
        self,
        prev_block: BlockRecord
    ) -> ExecutionPayloadV2:
        log.debug(f"Fetching execution payload for block: Bheight={prev_block.height}, Bhash={prev_block.header_hash}")
        
        self._ensure_web3_init()
        
        if self.peak_txb_hash != prev_block.header_hash:
            raise RuntimeError(f"Payload build on Bhash {self.peak_txb_hash} but requested {prev_block.header_hash}")
        
        if self.payload_id is None:
            raise RuntimeError("Execution payload was not built")

        try:
            raw_payload = self.w3.engine.get_payload_v2(self.payload_id).executionPayload
        except:
            self.connected = False
            raise
        self.connected = True
        
        transactions: List[bytes] = []
        for raw_transaction in raw_payload.transactions:
            transactions.append(hexstr_to_bytes(raw_transaction))
        
        withdrawals: List[WithdrawalV1] = []
        for raw_withdrawal in raw_payload.withdrawals:
            withdrawals.append(
                WithdrawalV1(
                    uint64(Web3.to_int(HexBytes(raw_withdrawal.index))),
                    uint64(Web3.to_int(HexBytes(raw_withdrawal.validatorIndex))),
                    bytes20.from_hexstr(raw_withdrawal.address),
                    uint64(Web3.to_int(HexBytes(raw_withdrawal.amount))),
                )
            )
        
        payload = ExecutionPayloadV2(
            bytes32.from_hexstr(raw_payload.parentHash),
            bytes20.from_hexstr(raw_payload.feeRecipient),
            bytes32.from_hexstr(raw_payload.stateRoot),
            bytes32.from_hexstr(raw_payload.receiptsRoot),
            bytes256.from_hexstr(raw_payload.logsBloom),
            bytes32.from_hexstr(raw_payload.prevRandao),
            uint64(Web3.to_int(HexBytes(raw_payload.blockNumber))),
            uint64(Web3.to_int(HexBytes(raw_payload.gasLimit))),
            uint64(Web3.to_int(HexBytes(raw_payload.gasUsed))),
            uint64(Web3.to_int(HexBytes(raw_payload.timestamp))),
            hexstr_to_bytes(raw_payload.extraData),
            uint256(Web3.to_int(HexBytes(raw_payload.baseFeePerGas))),
            bytes32.from_hexstr(raw_payload.blockHash),
            transactions,
            withdrawals,
        )
        
        log.info(
            f"Fetched execution payload: Bheight={prev_block.height}, Bhash={prev_block.header_hash}, "
            f"payload_id={self.payload_id}, Eheight={payload.blockNumber}, Ehash={payload.blockHash}"
        )
        
        return payload


    def _ensure_web3_init(self) -> None:
        if self.w3 is not None:
            return None
        
        execution_endpoint = self.beacon.config.get("execution_endpoint", "http://127.0.0.1:8551")
        
        selected_network = self.beacon.config.get("selected_network")
        if selected_network == "mainnet":
            default_secret_path = "../execution/geth/jwtsecret"
        else:
            default_secret_path = "../execution/" + selected_network + "/geth/jwtsecret"
        secret_path = path_from_root(
            self.beacon.root_path,
            self.beacon.config["network_overrides"]["config"][selected_network].get("jwt_secret", default_secret_path),
        )
        
        log.info(
            f"Initializing execution client connection to {execution_endpoint} "
            f"using JWT secret: {secret_path}"
        )

        try:
            secret_file = open(secret_path, 'r')
            secret = secret_file.readline()
            secret_file.close()
        except Exception as e:
            log.error(f"Exception in Web3 init: {e}")
            raise RuntimeError("Cannot open JWT secret file. Execution client is not running or needs more time to start.")
        
        self.w3 = Web3(
            HTTPAuthProvider(
                hexstr_to_bytes(secret),
                execution_endpoint,
            )
        )

        self.w3.attach_modules({
            "engine": EngineModule
        })
    
    
    def _create_payload_attributes(
        self,
        prev_tx_block: BlockRecord,
        coinbase: str,
    ) -> Dict[str, Any]:
        withdrawals = create_withdrawals(
            self.beacon.constants,
            prev_tx_block,
            self.beacon.blockchain,
        )
        raw_withdrawals = []
        
        for wd in withdrawals:
            raw_withdrawals.append({
                "index": Web3.to_hex(wd.index),
                "validatorIndex": Web3.to_hex(wd.validatorIndex),
                "address": "0x" + wd.address.hex(),
                "amount": Web3.to_hex(wd.amount)
            })
        
        return {
            "timestamp": Web3.to_hex(int(time.time())),
            "prevRandao": "0x" + prev_tx_block.reward_infusion_new_challenge.hex(),
            "suggestedFeeRecipient": coinbase,
            "withdrawals": raw_withdrawals,
        }
