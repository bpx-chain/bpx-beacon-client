from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from bpx.util.config import lock_and_load_config, save_config, str2bool


def configure(
    root_path: Path,
    set_farmer_peer: str,
    set_node_introducer: str,
    set_beacon_port: str,
    set_harvester_port: str,
    set_log_level: str,
    enable_upnp: str,
    set_outbound_peer_count: str,
    set_peer_count: str,
    testnet: str,
    peer_connect_timeout: str,
    crawler_db_path: str,
    crawler_minimum_version_count: Optional[int],
    seeder_domain_name: str,
    seeder_nameserver: str,
) -> None:
    config_yaml = "config.yaml"
    with lock_and_load_config(root_path, config_yaml, fill_missing_services=True) as config:
        change_made = False
        if set_node_introducer:
            try:
                if set_node_introducer.index(":"):
                    host, port = (
                        ":".join(set_node_introducer.split(":")[:-1]),
                        set_node_introducer.split(":")[-1],
                    )
                    config["beacon"]["introducer_peer"]["host"] = host
                    config["beacon"]["introducer_peer"]["port"] = int(port)
                    config["introducer"]["port"] = int(port)
                    print("Node introducer updated")
                    change_made = True
            except ValueError:
                print("Node introducer address must be in format [IP:Port]")
        if set_farmer_peer:
            try:
                if set_farmer_peer.index(":"):
                    host, port = (
                        ":".join(set_farmer_peer.split(":")[:-1]),
                        set_farmer_peer.split(":")[-1],
                    )
                    config["beacon"]["farmer_peer"]["host"] = host
                    config["beacon"]["farmer_peer"]["port"] = int(port)
                    config["harvester"]["farmer_peer"]["host"] = host
                    config["harvester"]["farmer_peer"]["port"] = int(port)
                    print("Farmer peer updated, make sure your harvester has the proper cert installed")
                    change_made = True
            except ValueError:
                print("Farmer address must be in format [IP:Port]")
        if set_beacon_port:
            config["beacon"]["port"] = int(set_beacon_port)
            config["farmer"]["beacon_peer"]["port"] = int(set_beacon_port)
            config["timelord"]["beacon_peer"]["port"] = int(set_beacon_port)
            print("Default beacon client port updated")
            change_made = True
        if set_harvester_port:
            config["harvester"]["port"] = int(set_harvester_port)
            config["farmer"]["harvester_peer"]["port"] = int(set_harvester_port)
            print("Default harvester port updated")
            change_made = True
        if set_log_level:
            levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]
            if set_log_level in levels:
                config["logging"]["log_level"] = set_log_level
                print(f"Logging level updated. Check {root_path}/log/debug.log")
                change_made = True
            else:
                print(f"Logging level not updated. Use one of: {levels}")
        if enable_upnp:
            config["beacon"]["enable_upnp"] = str2bool(enable_upnp)
            if str2bool(enable_upnp):
                print("uPnP enabled")
            else:
                print("uPnP disabled")
            change_made = True
        if set_outbound_peer_count:
            config["beacon"]["target_outbound_peer_count"] = int(set_outbound_peer_count)
            print("Target outbound peer count updated")
            change_made = True
        if set_peer_count:
            config["beacon"]["target_peer_count"] = int(set_peer_count)
            print("Target peer count updated")
            change_made = True
        if testnet:
            if testnet == "true" or testnet == "t":
                print("Setting Testnet")
                net_introducer = "introducer.testnet.bpxchain.cc"
                net_dns_introducer = "dns-introducer.testnet.bpxchain.cc"
                net_bootstrap_peers = ["node.testnet.bpxchain.cc"]
                net = "testnet"

            elif testnet == "false" or testnet == "f":
                print("Setting Mainnet")
                net_introducer = "introducer.mainnet.bpxchain.cc"
                net_dns_introducer = "dns-introducer.mainnet.bpxchain.cc"
                net_bootstrap_peers = ["node.mainnet.bpxchain.cc"]
                net = "mainnet"
                
            else:
                print("Please choose True or False")
                net = None
            
            if net is not None:
                net_port = "6201"
                net_introducer_port = "6202"
                
                # Set beacon port
                config["beacon"]["port"] = int(net_port)
                config["farmer"]["beacon_peer"]["port"] = int(net_port)
                config["timelord"]["beacon_peer"]["port"] = int(net_port)
                
                # Set introducer port
                config["beacon"]["introducer_peer"]["port"] = int(net_introducer_port)
                config["introducer"]["port"] = int(net_introducer_port)
                
                # Set introducer host
                config["beacon"]["introducer_peer"]["host"] = net_introducer
                
                # Set dns server
                config["beacon"]["dns_servers"] = [net_dns_introducer]
                
                # Set network
                config["selected_network"] = net
                config["harvester"]["selected_network"] = net
                config["farmer"]["selected_network"] = net
                config["timelord"]["selected_network"] = net
                config["beacon"]["selected_network"] = net
                config["ui"]["selected_network"] = net
                config["introducer"]["selected_network"] = net

                if "seeder" in config:
                    config["seeder"]["port"] = int(net_port)
                    config["seeder"]["other_peers_port"] = int(net_port)
                    config["seeder"]["selected_network"] = net
                    config["seeder"]["bootstrap_peers"] = net_bootstrap_peers

                print("Default beacon client port, introducer and network setting updated")
                change_made = True

        if peer_connect_timeout:
            config["beacon"]["peer_connect_timeout"] = int(peer_connect_timeout)
            change_made = True

        if crawler_db_path is not None and "seeder" in config:
            config["seeder"]["crawler_db_path"] = crawler_db_path
            change_made = True

        if crawler_minimum_version_count is not None and "seeder" in config:
            config["seeder"]["minimum_version_count"] = crawler_minimum_version_count
            change_made = True

        if seeder_domain_name is not None and "seeder" in config:
            config["seeder"]["domain_name"] = seeder_domain_name
            change_made = True

        if seeder_nameserver is not None and "seeder" in config:
            config["seeder"]["nameserver"] = seeder_nameserver
            change_made = True

        if change_made:
            print("Restart any running bpx services for changes to take effect")
            save_config(root_path, "config.yaml", config)


@click.command("configure", short_help="Modify configuration", no_args_is_help=True)
@click.option(
    "--testnet",
    "-t",
    help="configures for connection to testnet",
    type=click.Choice(["true", "t", "false", "f"]),
)
@click.option("--set-node-introducer", help="Set the introducer for node - IP:Port", type=str)
@click.option("--set-farmer-peer", help="Set the farmer peer for harvester - IP:Port", type=str)
@click.option(
    "--set-beacon-port",
    help="Set the port to use for the beacon, useful for testing",
    type=str,
)
@click.option(
    "--set-harvester-port",
    help="Set the port to use for the harvester, useful for testing",
    type=str,
)
@click.option(
    "--set-log-level",
    "--log-level",
    "-log-level",
    help="Set the instance log level",
    type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]),
)
@click.option(
    "--enable-upnp",
    "--upnp",
    "-upnp",
    help="Enable or disable uPnP",
    type=click.Choice(["true", "t", "false", "f"]),
)
@click.option(
    "--set_outbound-peer-count",
    help="Update the target outbound peer count (default 8)",
    type=str,
)
@click.option("--set-peer-count", help="Update the target peer count (default 80)", type=str)
@click.option("--set-peer-connect-timeout", help="Update the peer connect timeout (default 30)", type=str)
@click.option(
    "--crawler-db-path",
    help="configures the path to the crawler database",
    type=str,
)
@click.option(
    "--crawler-minimum-version-count",
    help="configures how many of a particular version must be seen to be reported in logs",
    type=int,
)
@click.option(
    "--seeder-domain-name",
    help="configures the seeder domain_name setting. Ex: `seeder.example.com.`",
    type=str,
)
@click.option(
    "--seeder-nameserver",
    help="configures the seeder nameserver setting. Ex: `example.com.`",
    type=str,
)
@click.pass_context
def configure_cmd(
    ctx: click.Context,
    set_farmer_peer: str,
    set_node_introducer: str,
    set_beacon_port: str,
    set_harvester_port: str,
    set_log_level: str,
    enable_upnp: str,
    set_outbound_peer_count: str,
    set_peer_count: str,
    testnet: str,
    set_peer_connect_timeout: str,
    crawler_db_path: str,
    crawler_minimum_version_count: int,
    seeder_domain_name: str,
    seeder_nameserver: str,
) -> None:
    configure(
        ctx.obj["root_path"],
        set_farmer_peer,
        set_node_introducer,
        set_beacon_port,
        set_harvester_port,
        set_log_level,
        enable_upnp,
        set_outbound_peer_count,
        set_peer_count,
        testnet,
        set_peer_connect_timeout,
        crawler_db_path,
        crawler_minimum_version_count,
        seeder_domain_name,
        seeder_nameserver,
    )
